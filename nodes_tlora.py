import logging
import math
import os
import re
import sys
import threading
import uuid

import folder_paths
import torch
import torch.nn.functional as F

import comfy.lora
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter


_TLORA_CONFIG_ATTACHMENT = "tlora_runtime_config"
_TLORA_PREDICT_WRAPPER_KEY = "tlora_predict_noise_mask_wrapper"
_TLORA_INJECTION_KEY_PREFIX = "tlora_bypass"
_LYCORIS_TLORA_CONFIG_ATTACHMENT = "lycoris_tlora_runtime_config"
_LYCORIS_TLORA_PREDICT_WRAPPER_KEY = "lycoris_tlora_predict_noise_wrapper"
_LYCORIS_TLORA_INJECTION_KEY_PREFIX = "lycoris_tlora_bypass"
_TLORA_KEY_PATTERN = re.compile(
    r"^(?:unet\.)?(?P<base>.+?\.to_(?:q|k|v|out(?:\.0)?|q_proj|k_proj|v_proj))_lora\.(?P<param>.+)$"
)
_LYCORIS_RUNTIME_CACHE = None

_TLORA_STATE = threading.local()


def _set_tlora_mask(mask: torch.Tensor):
    _TLORA_STATE.mask = mask


def _get_tlora_mask():
    return getattr(_TLORA_STATE, "mask", None)


def _clear_tlora_mask():
    if hasattr(_TLORA_STATE, "mask"):
        delattr(_TLORA_STATE, "mask")
    if hasattr(_TLORA_STATE, "debug_step"):
        delattr(_TLORA_STATE, "debug_step")
    if hasattr(_TLORA_STATE, "adapter_log_emitted"):
        delattr(_TLORA_STATE, "adapter_log_emitted")


def _clamp_int(value: int, min_value: int, max_value: int):
    return max(min_value, min(max_value, value))


def _compute_active_rank(
    timestep: float,
    max_timestep: int,
    max_rank: int,
    min_rank: int,
    alpha: float,
):
    if max_rank <= 0:
        return 0

    min_rank = _clamp_int(min_rank, 0, max_rank)
    if max_timestep <= 0:
        return min_rank

    t = float(timestep)
    if not math.isfinite(t):
        t = float(max_timestep)
    t = max(0.0, min(float(max_timestep), t))

    progress = (float(max_timestep) - t) / float(max_timestep)
    progress = max(0.0, min(1.0, progress))
    progress = progress ** float(alpha)

    active_rank = int(progress * (max_rank - min_rank)) + min_rank
    return _clamp_int(active_rank, 0, max_rank)


def _rank_mask_tensor(active_rank: int, max_rank: int, device: torch.device, dtype: torch.dtype):
    mask = torch.zeros((1, max_rank), device=device, dtype=dtype)
    if active_rank > 0:
        mask[:, :active_rank] = 1.0
    return mask


def _extract_sigma_scalar(timestep):
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return None
        return float(timestep.reshape(-1)[0].detach().float().cpu())
    try:
        return float(timestep)
    except Exception:
        return None


def _sigma_to_timestep(model_sampling, sigma_value: float):
    if model_sampling is None or sigma_value is None:
        return None
    if not hasattr(model_sampling, "timestep"):
        return sigma_value

    try:
        sigma = torch.tensor([sigma_value], dtype=torch.float32)
        t = model_sampling.timestep(sigma)
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            return float(t.reshape(-1)[0].detach().float().cpu())
    except Exception:
        return sigma_value
    return sigma_value


def _resolve_max_timestep(model_patcher, requested_max_timestep: int):
    if requested_max_timestep is not None and int(requested_max_timestep) > 0:
        return int(requested_max_timestep)

    model_sampling = model_patcher.get_model_object("model_sampling")
    for attr_name in ("num_timesteps", "multiplier"):
        if hasattr(model_sampling, attr_name):
            value = int(getattr(model_sampling, attr_name))
            if value > 0:
                return value

    return 1000


def _import_lycoris_runtime():
    global _LYCORIS_RUNTIME_CACHE
    if _LYCORIS_RUNTIME_CACHE is not None:
        return _LYCORIS_RUNTIME_CACHE

    tried_paths = []
    import_error = None
    search_paths = []
    env_path = os.environ.get("LYCORIS_PATH", None)
    if env_path:
        search_paths.append(os.path.expanduser(env_path))
    search_paths.extend(
        [
            os.path.expanduser("~/src/LyCORIS"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LyCORIS")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "LyCORIS")),
        ]
    )

    for candidate in ("", *search_paths):
        if candidate:
            if not os.path.isdir(candidate):
                continue
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            tried_paths.append(candidate)
        try:
            import lycoris

            required_attrs = (
                "create_lycoris_from_weights",
                "TLoraModule",
                "set_timestep_mask",
                "get_timestep_mask",
                "clear_timestep_mask",
                "compute_timestep_mask",
            )
            missing = [name for name in required_attrs if not hasattr(lycoris, name)]
            if missing:
                raise ImportError(
                    "Installed lycoris package is missing T-LoRA APIs: {}. "
                    "Install lycoris-lora from LyCORIS git main."
                    .format(", ".join(missing))
                )

            _LYCORIS_RUNTIME_CACHE = {
                "create_lycoris_from_weights": lycoris.create_lycoris_from_weights,
                "TLoraModule": lycoris.TLoraModule,
                "set_timestep_mask": lycoris.set_timestep_mask,
                "get_timestep_mask": lycoris.get_timestep_mask,
                "clear_timestep_mask": lycoris.clear_timestep_mask,
                "compute_timestep_mask": lycoris.compute_timestep_mask,
            }
            return _LYCORIS_RUNTIME_CACHE
        except Exception as exc:
            import_error = exc

    raise ImportError(
        "LyCORIS could not be imported with required T-LoRA APIs. "
        "Install lycoris-lora from git main, e.g. "
        "`pip install -U git+https://github.com/KohakuBlueleaf/LyCORIS.git`, "
        "or set LYCORIS_PATH (tried: {})".format(tried_paths)
    ) from import_error


def _prepare_rank_mask(rank: int, reference: torch.Tensor):
    current_mask = _get_tlora_mask()
    if current_mask is None:
        return None

    mask = current_mask.to(device=reference.device)
    if mask.ndim == 1:
        mask = mask.view(1, -1)
    elif mask.ndim != 2:
        mask = mask.reshape(1, -1)

    if mask.shape[1] < rank:
        pad = torch.ones((1, rank - mask.shape[1]), device=mask.device, dtype=mask.dtype)
        mask = torch.cat([mask, pad], dim=1)
    elif mask.shape[1] > rank:
        mask = mask[:, :rank]

    return mask


def _prepare_lycoris_rank_mask(
    rank: int,
    reference: torch.Tensor,
    get_timestep_mask,
    group_id: int,
):
    mask = get_timestep_mask(int(group_id))
    if mask is None:
        mask = torch.ones((1, rank), device=reference.device, dtype=reference.dtype)
        return mask

    mask = mask.to(device=reference.device)
    if mask.ndim == 1:
        mask = mask.view(1, -1)
    elif mask.ndim != 2:
        mask = mask.reshape(1, -1)

    if mask.shape[1] < rank:
        pad = torch.ones((1, rank - mask.shape[1]), device=mask.device, dtype=mask.dtype)
        mask = torch.cat([mask, pad], dim=1)
    elif mask.shape[1] > rank:
        mask = mask[:, :rank]

    return mask.to(device=reference.device, dtype=reference.dtype)


def _maybe_log_adapter_mask(adapter_name: str, rank: int, mask: torch.Tensor):
    if getattr(_TLORA_STATE, "adapter_log_emitted", False):
        return
    debug_step = getattr(_TLORA_STATE, "debug_step", None)
    if debug_step is None:
        return
    logging.info(
        "[ComfyUI-T-LoRA][adapter] step=%s adapter=%s rank=%d active=%d",
        debug_step.get("step"),
        adapter_name,
        rank,
        int(mask.sum().item()),
    )
    _TLORA_STATE.adapter_log_emitted = True


class _TLoraAdapterBase:
    def __init__(self, rank: int, weights: tuple[torch.Tensor, ...]):
        self.rank = int(rank)
        self.weights = tuple(weights)
        self.multiplier = 1.0

    def g(self, y: torch.Tensor):
        return y


class _TLoraLinearAdapter(_TLoraAdapterBase):
    def __init__(self, down: torch.Tensor, up: torch.Tensor):
        super().__init__(rank=down.shape[0], weights=(up, down))

    def h(self, x: torch.Tensor, _base_out: torch.Tensor):
        up, down = self.weights
        orig_dtype = x.dtype
        dtype = down.dtype
        x_cast = x.to(dtype)

        if down.device != x_cast.device:
            down = down.to(device=x_cast.device)
        if up.device != x_cast.device:
            up = up.to(device=x_cast.device)

        down_hidden = F.linear(x_cast, down)

        rank_mask = _prepare_rank_mask(self.rank, down_hidden)
        if rank_mask is not None:
            _maybe_log_adapter_mask("linear", self.rank, rank_mask)
            down_hidden = down_hidden * rank_mask.to(
                device=down_hidden.device, dtype=down_hidden.dtype
            )

        up_hidden = F.linear(down_hidden, up)
        return up_hidden.to(orig_dtype) * float(self.multiplier)


class _TLoraOrthogonalAdapter(_TLoraAdapterBase):
    def __init__(
        self,
        q_layer: torch.Tensor,
        p_layer: torch.Tensor,
        lambda_layer: torch.Tensor,
        base_q: torch.Tensor,
        base_p: torch.Tensor,
        base_lambda: torch.Tensor,
    ):
        super().__init__(
            rank=q_layer.shape[0],
            weights=(q_layer, p_layer, lambda_layer, base_q, base_p, base_lambda),
        )

    def h(self, x: torch.Tensor, _base_out: torch.Tensor):
        q_layer, p_layer, lambda_layer, base_q, base_p, base_lambda = self.weights
        orig_dtype = x.dtype
        dtype = q_layer.dtype
        x_cast = x.to(dtype)

        rank_mask = _prepare_rank_mask(self.rank, x_cast)
        if rank_mask is None:
            rank_mask = torch.ones((1, self.rank), device=x_cast.device, dtype=x_cast.dtype)
        else:
            rank_mask = rank_mask.to(device=x_cast.device, dtype=x_cast.dtype)
        _maybe_log_adapter_mask("orthogonal", self.rank, rank_mask)

        if q_layer.device != x_cast.device:
            q_layer = q_layer.to(device=x_cast.device)
        if p_layer.device != x_cast.device:
            p_layer = p_layer.to(device=x_cast.device)
        if base_q.device != x_cast.device:
            base_q = base_q.to(device=x_cast.device)
        if base_p.device != x_cast.device:
            base_p = base_p.to(device=x_cast.device)
        if lambda_layer.device != x_cast.device:
            lambda_layer = lambda_layer.to(device=x_cast.device)
        if base_lambda.device != x_cast.device:
            base_lambda = base_lambda.to(device=x_cast.device)
        if lambda_layer.dtype != x_cast.dtype:
            lambda_layer = lambda_layer.to(dtype=x_cast.dtype)
        if base_lambda.dtype != x_cast.dtype:
            base_lambda = base_lambda.to(dtype=x_cast.dtype)

        q_hidden = F.linear(x_cast, q_layer) * lambda_layer * rank_mask
        p_hidden = F.linear(q_hidden, p_layer)

        base_hidden = F.linear(x_cast, base_q) * base_lambda * rank_mask
        base_out = F.linear(base_hidden, base_p)

        result = p_hidden - base_out
        return result.to(orig_dtype) * float(self.multiplier)


class _TLoraSegmentedLinearAdapter(_TLoraAdapterBase):
    """
    Adapter for fused linear projections (e.g. Flux qkv projections).

    Each segment contributes a masked LoRA delta into a slice of the module output.
    """

    def __init__(self, segments: list[dict]):
        max_rank = 0
        weights = []
        for segment in segments:
            down = segment["down"]
            up = segment["up"]
            max_rank = max(max_rank, int(down.shape[0]))
            weights.extend([down, up])
        super().__init__(rank=max_rank, weights=tuple(weights))
        self.segments = list(segments)

    def h(self, x: torch.Tensor, base_out: torch.Tensor):
        orig_dtype = x.dtype

        if len(self.segments) == 0:
            return torch.zeros_like(base_out)

        x_dtype = self.segments[0]["down"].dtype
        x_cast = x.to(x_dtype)
        delta = torch.zeros_like(base_out, dtype=x_cast.dtype)
        logged = False

        for segment in self.segments:
            down = segment["down"]
            up = segment["up"]
            offset = int(segment["offset"])
            length = int(segment["length"])

            if down.device != x_cast.device:
                down = down.to(device=x_cast.device)
            if up.device != x_cast.device:
                up = up.to(device=x_cast.device)
            if down.dtype != x_cast.dtype:
                down = down.to(dtype=x_cast.dtype)
            if up.dtype != x_cast.dtype:
                up = up.to(dtype=x_cast.dtype)

            down_hidden = F.linear(x_cast, down)
            rank_mask = _prepare_rank_mask(int(down.shape[0]), down_hidden)
            if rank_mask is not None:
                if not logged:
                    _maybe_log_adapter_mask("segmented", int(down.shape[0]), rank_mask)
                    logged = True
                down_hidden = down_hidden * rank_mask.to(
                    device=down_hidden.device, dtype=down_hidden.dtype
                )

            up_hidden = F.linear(down_hidden, up)
            end = min(offset + length, delta.shape[-1])
            if end > offset:
                seg_width = end - offset
                delta[..., offset:end] = delta[..., offset:end] + up_hidden[..., :seg_width]

        return delta.to(orig_dtype) * float(self.multiplier)


class _LycorisTLoraAdapter(_TLoraAdapterBase):
    def __init__(
        self,
        q_layer: torch.Tensor,
        p_layer: torch.Tensor,
        lambda_layer: torch.Tensor,
        base_q: torch.Tensor,
        base_p: torch.Tensor,
        base_lambda: torch.Tensor,
        scale: float,
        get_timestep_mask,
        mask_group_id: int,
        is_conv: bool,
        kw_dict_down: dict,
        kw_dict_up: dict,
    ):
        super().__init__(
            rank=q_layer.shape[0],
            weights=(q_layer, p_layer, lambda_layer, base_q, base_p, base_lambda),
        )
        self.scale = float(scale)
        self.get_timestep_mask = get_timestep_mask
        self.mask_group_id = int(mask_group_id)
        self.is_conv = bool(is_conv)
        self.kw_dict_down = dict(kw_dict_down or {})
        self.kw_dict_up = dict(kw_dict_up or {})

    @staticmethod
    def _conv_op(weight: torch.Tensor):
        if weight.ndim == 3:
            return F.conv1d
        if weight.ndim == 4:
            return F.conv2d
        if weight.ndim == 5:
            return F.conv3d
        raise ValueError(f"Unsupported convolution weight rank for T-LoRA: ndim={weight.ndim}")

    def h(self, x: torch.Tensor, _base_out: torch.Tensor):
        q_layer, p_layer, lambda_layer, base_q, base_p, base_lambda = self.weights
        orig_dtype = x.dtype
        dtype = q_layer.dtype
        x_cast = x.to(dtype)

        if q_layer.device != x_cast.device:
            q_layer = q_layer.to(device=x_cast.device)
        if p_layer.device != x_cast.device:
            p_layer = p_layer.to(device=x_cast.device)
        if base_q.device != x_cast.device:
            base_q = base_q.to(device=x_cast.device)
        if base_p.device != x_cast.device:
            base_p = base_p.to(device=x_cast.device)
        if lambda_layer.device != x_cast.device:
            lambda_layer = lambda_layer.to(device=x_cast.device)
        if base_lambda.device != x_cast.device:
            base_lambda = base_lambda.to(device=x_cast.device)

        q_layer = q_layer.to(dtype=x_cast.dtype)
        p_layer = p_layer.to(dtype=x_cast.dtype)
        base_q = base_q.to(dtype=x_cast.dtype)
        base_p = base_p.to(dtype=x_cast.dtype)
        lambda_layer = lambda_layer.to(dtype=x_cast.dtype)
        base_lambda = base_lambda.to(dtype=x_cast.dtype)

        rank_mask = _prepare_lycoris_rank_mask(
            self.rank, x_cast, self.get_timestep_mask, self.mask_group_id
        )
        _maybe_log_adapter_mask("lycoris", self.rank, rank_mask)

        lam = lambda_layer * rank_mask
        lam_base = base_lambda * rank_mask

        if self.is_conv:
            conv_op = self._conv_op(q_layer)

            q_out = conv_op(x_cast, q_layer, None, **self.kw_dict_down)
            q_out_scaled = q_out * lam.view(1, -1, *([1] * (q_out.dim() - 2)))
            curr_out = conv_op(q_out_scaled, p_layer, None, **self.kw_dict_up)

            q_base_out = conv_op(x_cast, base_q, None, **self.kw_dict_down)
            q_base_scaled = q_base_out * lam_base.view(1, -1, *([1] * (q_base_out.dim() - 2)))
            base_out = conv_op(q_base_scaled, base_p, None, **self.kw_dict_up)
        else:
            q_out = F.linear(x_cast, q_layer)
            q_out_scaled = q_out * lam
            curr_out = F.linear(q_out_scaled, p_layer)

            q_base_out = F.linear(x_cast, base_q)
            q_base_scaled = q_base_out * lam_base
            base_out = F.linear(q_base_scaled, base_p)

        result = (curr_out - base_out) * self.scale
        return result.to(orig_dtype) * float(self.multiplier)


def _candidate_base_keys(base_key: str):
    candidates = [base_key]

    # Diffusers SDXL attention processor-style naming.
    if ".to_out" in base_key and ".to_out.0" not in base_key:
        candidates.append(base_key.replace(".to_out", ".to_out.0"))
    if ".to_out.0" in base_key:
        candidates.append(base_key.replace(".to_out.0", ".to_out"))

    # Flux/SD3 attention processor keys often include ".processor." while Comfy's
    # key map usually targets module paths without processor in the key.
    without_processor = [x.replace(".processor.", ".") for x in candidates if ".processor." in x]
    candidates.extend(without_processor)

    # Flux context projection naming in T-LoRA uses to_*_proj while Comfy maps
    # add_*_proj from the underlying attention module.
    proj_map = {
        ".to_q_proj": ".add_q_proj",
        ".to_k_proj": ".add_k_proj",
        ".to_v_proj": ".add_v_proj",
    }
    expanded = list(candidates)
    for key in expanded:
        for src, dst in proj_map.items():
            if src in key:
                candidates.append(key.replace(src, dst))

    # Preserve order while deduplicating.
    return list(dict.fromkeys(candidates))


def _resolve_unet_key(base_key: str, key_map: dict[str, str]):
    candidates = _candidate_base_keys(base_key)

    for candidate in candidates:
        mapped = key_map.get(candidate, None)
        if mapped is not None:
            return mapped
        mapped = key_map.get(f"unet.{candidate}", None)
        if mapped is not None:
            return mapped
    return None


def _normalize_mapped_key(mapped_key):
    """
    Normalize mapped keys from comfy.lora key maps.

    Returns:
      - (target_weight_key: str, slice_spec: tuple|None)
      - slice_spec is (dim, offset, length) for fused projections.
    """
    if isinstance(mapped_key, str):
        return mapped_key, None

    if (
        isinstance(mapped_key, tuple)
        and len(mapped_key) >= 2
        and isinstance(mapped_key[0], str)
        and isinstance(mapped_key[1], tuple)
        and len(mapped_key[1]) >= 3
    ):
        dim = int(mapped_key[1][0])
        offset = int(mapped_key[1][1])
        length = int(mapped_key[1][2])
        return mapped_key[0], (dim, offset, length)

    return None, None


def _should_ignore_unmapped_tlora_key(base_key: str):
    # Flux single-transformer blocks do not expose add_*_proj in Comfy's key map.
    # Upstream T-LoRA still serializes these unused processor params.
    if base_key.startswith("single_transformer_blocks.") and any(
        x in base_key for x in (".to_q_proj", ".to_k_proj", ".to_v_proj")
    ):
        return True
    return False


def _group_tlora_state_dict(state_dict: dict, key_map: dict[str, str]):
    grouped = {}
    skipped = []

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        match = _TLORA_KEY_PATTERN.match(key)
        if match is None:
            continue

        base_key = match.group("base")
        param_name = match.group("param")
        model_weight_key = _resolve_unet_key(base_key, key_map)
        if model_weight_key is None:
            if not _should_ignore_unmapped_tlora_key(base_key):
                skipped.append(key)
            continue

        target_key, slice_spec = _normalize_mapped_key(model_weight_key)
        if target_key is None:
            if not _should_ignore_unmapped_tlora_key(base_key):
                skipped.append(key)
            continue

        grouped.setdefault((target_key, slice_spec, base_key), {})[param_name] = value

    return grouped, skipped


def _build_tlora_adapter(params: dict):
    if "down.weight" in params and "up.weight" in params:
        return _TLoraLinearAdapter(
            down=params["down.weight"],
            up=params["up.weight"],
        )

    required_ortho = {
        "q_layer.weight",
        "p_layer.weight",
        "lambda_layer",
        "base_q.weight",
        "base_p.weight",
        "base_lambda",
    }
    if required_ortho.issubset(params.keys()):
        return _TLoraOrthogonalAdapter(
            q_layer=params["q_layer.weight"],
            p_layer=params["p_layer.weight"],
            lambda_layer=params["lambda_layer"],
            base_q=params["base_q.weight"],
            base_p=params["base_p.weight"],
            base_lambda=params["base_lambda"],
        )

    return None


def _load_tlora_bypass_for_model(model, state_dict, strength_model):
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    grouped, skipped = _group_tlora_state_dict(state_dict, key_map)

    model_lora = model.clone()
    manager = comfy.weight_adapter.BypassInjectionManager()
    model_sd_keys = set(model_lora.model.state_dict().keys())
    loaded_count = 0
    max_rank_in_ckpt = 0
    segmented_targets = {}

    for group_key, params in grouped.items():
        model_weight_key, slice_spec, source_base_key = group_key
        adapter = _build_tlora_adapter(params)
        if slice_spec is not None:
            # Fused target: accumulate linear (down/up) adapters into output slices.
            if adapter is None or not isinstance(adapter, _TLoraLinearAdapter):
                logging.warning(
                    "[ComfyUI-T-LoRA] Unsupported sliced adapter params for %s (%s): %s",
                    model_weight_key,
                    source_base_key,
                    sorted(params.keys()),
                )
                continue

            dim, offset, length = slice_spec
            if int(dim) != 0:
                logging.warning(
                    "[ComfyUI-T-LoRA] Unsupported slice dim=%d for %s (%s)",
                    int(dim),
                    model_weight_key,
                    source_base_key,
                )
                continue

            up, down = adapter.weights
            segmented_targets.setdefault(model_weight_key, []).append(
                {
                    "down": down,
                    "up": up,
                    "offset": int(offset),
                    "length": int(length),
                    "source": source_base_key,
                }
            )
            max_rank_in_ckpt = max(max_rank_in_ckpt, adapter.rank)
            continue

        if adapter is None:
            logging.warning(
                "[ComfyUI-T-LoRA] Unsupported adapter param set for %s: %s",
                model_weight_key,
                sorted(params.keys()),
            )
            continue

        if model_weight_key not in model_sd_keys:
            logging.warning(
                "[ComfyUI-T-LoRA] Target key missing in model: %s",
                model_weight_key,
            )
            continue

        manager.add_adapter(model_weight_key, adapter, strength=strength_model)
        loaded_count += 1
        max_rank_in_ckpt = max(max_rank_in_ckpt, adapter.rank)

    for model_weight_key, segments in segmented_targets.items():
        if model_weight_key not in model_sd_keys:
            logging.warning(
                "[ComfyUI-T-LoRA] Fused target key missing in model: %s",
                model_weight_key,
            )
            continue

        segments = sorted(segments, key=lambda x: (int(x["offset"]), int(x["length"])))
        adapter = _TLoraSegmentedLinearAdapter(segments=segments)
        manager.add_adapter(model_weight_key, adapter, strength=strength_model)
        loaded_count += 1
        max_rank_in_ckpt = max(max_rank_in_ckpt, adapter.rank)

    for key in skipped:
        logging.warning("[ComfyUI-T-LoRA] Unmapped T-LoRA key: %s", key)

    injections = manager.create_injections(model_lora.model)
    if manager.get_hook_count() <= 0:
        raise ValueError("No T-LoRA adapters were created from this checkpoint.")

    injection_key = f"{_TLORA_INJECTION_KEY_PREFIX}_{uuid.uuid4().hex}"
    model_lora.set_injections(injection_key, injections)
    return model_lora, loaded_count, max_rank_in_ckpt


def _build_module_weight_key_map(model_root):
    module_weight_map = {}
    model_sd_keys = set(model_root.state_dict().keys())
    for module_name, module in model_root.named_modules():
        if not module_name:
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        key = f"{module_name}.weight"
        if key in model_sd_keys:
            module_weight_map[id(module)] = key
    return module_weight_map


def _create_lycoris_network(lycoris_runtime, state_dict, module_root):
    create_lycoris_from_weights = lycoris_runtime["create_lycoris_from_weights"]
    network, _ = create_lycoris_from_weights(
        multiplier=1.0,
        file="",
        module=module_root,
        weights_sd=state_dict,
    )
    return network


def _load_lycoris_tlora_bypass_for_model(
    model,
    state_dict,
    strength_model,
    mask_group_id,
    lycoris_runtime,
):
    model_lora = model.clone()
    module_weight_map = _build_module_weight_key_map(model_lora.model)

    root_candidates = []
    diffusion_model = getattr(model_lora.model, "diffusion_model", None)
    if diffusion_model is not None:
        root_candidates.append(("diffusion_model", diffusion_model))
    root_candidates.append(("model", model_lora.model))

    best_network = None
    best_count = 0
    last_error = None
    tlora_module_type = lycoris_runtime["TLoraModule"]

    for label, root in root_candidates:
        try:
            network = _create_lycoris_network(lycoris_runtime, state_dict, root)
        except Exception as exc:
            last_error = exc
            logging.warning(
                "[ComfyUI-T-LoRA] LyCORIS network build failed on %s root: %s",
                label,
                exc,
            )
            continue

        tlora_count = sum(1 for lora in network.loras if isinstance(lora, tlora_module_type))
        if tlora_count > best_count:
            best_count = tlora_count
            best_network = network

    if best_network is None:
        raise ValueError(
            "Could not construct a LyCORIS network from this checkpoint."
        ) from last_error

    manager = comfy.weight_adapter.BypassInjectionManager()
    loaded_count = 0
    max_rank_in_ckpt = 0
    skipped_non_tlora = 0

    get_timestep_mask = lycoris_runtime["get_timestep_mask"]
    for lora_module in best_network.loras:
        if not isinstance(lora_module, tlora_module_type):
            skipped_non_tlora += 1
            continue

        target_module = lora_module.org_module[0]
        model_weight_key = module_weight_map.get(id(target_module), None)
        if model_weight_key is None:
            logging.warning(
                "[ComfyUI-T-LoRA] Could not map LyCORIS module to model weight: %s",
                lora_module.lora_name,
            )
            continue

        rank = int(getattr(lora_module, "lora_dim", 0))
        if rank <= 0:
            logging.warning(
                "[ComfyUI-T-LoRA] Invalid LyCORIS T-LoRA rank for %s",
                lora_module.lora_name,
            )
            continue

        # Keep deterministic inference behavior.
        lora_module.eval()
        lora_module.mask_group_id = int(mask_group_id)

        adapter = _LycorisTLoraAdapter(
            q_layer=lora_module.q_layer.weight.detach().clone(),
            p_layer=lora_module.p_layer.weight.detach().clone(),
            lambda_layer=lora_module.lambda_layer.detach().clone(),
            base_q=lora_module.base_q.detach().clone(),
            base_p=lora_module.base_p.detach().clone(),
            base_lambda=lora_module.base_lambda.detach().clone(),
            scale=float(lora_module.scale),
            get_timestep_mask=get_timestep_mask,
            mask_group_id=mask_group_id,
            is_conv=bool(getattr(lora_module, "isconv", False)),
            kw_dict_down=getattr(lora_module, "kw_dict_down", {}),
            kw_dict_up=getattr(lora_module, "kw_dict_up", {}),
        )

        manager.add_adapter(model_weight_key, adapter, strength=strength_model)
        loaded_count += 1
        max_rank_in_ckpt = max(max_rank_in_ckpt, rank)

    if loaded_count <= 0:
        raise ValueError(
            "No LyCORIS T-LoRA adapters were created from this checkpoint."
        )

    if skipped_non_tlora > 0:
        logging.warning(
            "[ComfyUI-T-LoRA] Skipped %d non-TLoRA LyCORIS modules in checkpoint.",
            skipped_non_tlora,
        )

    injections = manager.create_injections(model_lora.model)
    injection_key = f"{_LYCORIS_TLORA_INJECTION_KEY_PREFIX}_{uuid.uuid4().hex}"
    model_lora.set_injections(injection_key, injections)
    return model_lora, loaded_count, max_rank_in_ckpt


def _tlora_predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    model_options = model_options or {}
    model_patcher = executor.class_obj.model_patcher
    config = model_patcher.get_attachment(_TLORA_CONFIG_ATTACHMENT)
    if config is None:
        return executor(x, timestep, model_options=model_options, seed=seed)

    sigma = _extract_sigma_scalar(timestep)
    model_sampling = model_patcher.get_model_object("model_sampling")
    t_value = _sigma_to_timestep(model_sampling, sigma)

    if t_value is None:
        return executor(x, timestep, model_options=model_options, seed=seed)

    active_rank = _compute_active_rank(
        timestep=t_value,
        max_timestep=config["max_timestep"],
        max_rank=config["max_rank"],
        min_rank=config["min_rank"],
        alpha=config["alpha"],
    )
    config["step_counter"] = int(config.get("step_counter", 0)) + 1
    step_counter = int(config["step_counter"])

    if bool(config.get("debug", False)):
        debug_every = max(1, int(config.get("debug_every", 1)))
        if step_counter == 1 or step_counter % debug_every == 0:
            logging.info(
                "[ComfyUI-T-LoRA][step] step=%d sigma=%.6f timestep=%.3f active_rank=%d/%d min_rank=%d alpha=%.3f max_timestep=%d",
                step_counter,
                float(sigma),
                float(t_value),
                int(active_rank),
                int(config["max_rank"]),
                int(config["min_rank"]),
                float(config["alpha"]),
                int(config["max_timestep"]),
            )

    step_mask = _rank_mask_tensor(
        active_rank=active_rank,
        max_rank=config["max_rank"],
        device=x.device,
        dtype=x.dtype,
    )

    _set_tlora_mask(step_mask)
    _TLORA_STATE.debug_step = {"step": step_counter}
    _TLORA_STATE.adapter_log_emitted = False
    try:
        return executor(x, timestep, model_options=model_options, seed=seed)
    finally:
        _clear_tlora_mask()


def _configure_tlora_runtime(model_patcher, max_rank, min_rank, alpha, max_timestep, debug=False, debug_every=1):
    resolved_max_timestep = _resolve_max_timestep(model_patcher, max_timestep)

    config = {
        "max_rank": int(max_rank),
        "min_rank": int(min_rank),
        "alpha": float(alpha),
        "max_timestep": int(resolved_max_timestep),
        "debug": bool(debug),
        "debug_every": int(debug_every),
        "step_counter": 0,
    }
    model_patcher.set_attachments(_TLORA_CONFIG_ATTACHMENT, config)
    model_patcher.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        _TLORA_PREDICT_WRAPPER_KEY,
    )
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        _TLORA_PREDICT_WRAPPER_KEY,
        _tlora_predict_noise_wrapper,
    )


def _lycoris_tlora_predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    model_options = model_options or {}
    model_patcher = executor.class_obj.model_patcher
    config = model_patcher.get_attachment(_LYCORIS_TLORA_CONFIG_ATTACHMENT)
    if config is None:
        return executor(x, timestep, model_options=model_options, seed=seed)

    sigma = _extract_sigma_scalar(timestep)
    model_sampling = model_patcher.get_model_object("model_sampling")
    t_value = _sigma_to_timestep(model_sampling, sigma)

    if t_value is None:
        return executor(x, timestep, model_options=model_options, seed=seed)

    active_rank = _compute_active_rank(
        timestep=t_value,
        max_timestep=config["max_timestep"],
        max_rank=config["max_rank"],
        min_rank=config["min_rank"],
        alpha=config["alpha"],
    )
    config["step_counter"] = int(config.get("step_counter", 0)) + 1
    step_counter = int(config["step_counter"])

    if bool(config.get("debug", False)):
        debug_every = max(1, int(config.get("debug_every", 1)))
        if step_counter == 1 or step_counter % debug_every == 0:
            logging.info(
                "[ComfyUI-T-LoRA][step][lycoris] step=%d sigma=%.6f timestep=%.3f active_rank=%d/%d min_rank=%d alpha=%.3f max_timestep=%d group=%d",
                step_counter,
                float(sigma),
                float(t_value),
                int(active_rank),
                int(config["max_rank"]),
                int(config["min_rank"]),
                float(config["alpha"]),
                int(config["max_timestep"]),
                int(config["mask_group_id"]),
            )

    mask = config["compute_timestep_mask"](
        timestep=int(round(float(t_value))),
        max_timestep=int(config["max_timestep"]),
        max_rank=int(config["max_rank"]),
        min_rank=int(config["min_rank"]),
        alpha=float(config["alpha"]),
    )
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    mask = mask.to(device=x.device, dtype=x.dtype)

    config["set_timestep_mask"](mask, int(config["mask_group_id"]))
    _TLORA_STATE.debug_step = {"step": step_counter}
    _TLORA_STATE.adapter_log_emitted = False
    try:
        return executor(x, timestep, model_options=model_options, seed=seed)
    finally:
        config["clear_timestep_mask"](int(config["mask_group_id"]))
        _clear_tlora_mask()


def _configure_lycoris_tlora_runtime(
    model_patcher,
    lycoris_runtime,
    max_rank,
    min_rank,
    alpha,
    max_timestep,
    mask_group_id=0,
    debug=False,
    debug_every=1,
):
    resolved_max_timestep = _resolve_max_timestep(model_patcher, max_timestep)

    config = {
        "max_rank": int(max_rank),
        "min_rank": int(min_rank),
        "alpha": float(alpha),
        "max_timestep": int(resolved_max_timestep),
        "mask_group_id": int(mask_group_id),
        "set_timestep_mask": lycoris_runtime["set_timestep_mask"],
        "clear_timestep_mask": lycoris_runtime["clear_timestep_mask"],
        "compute_timestep_mask": lycoris_runtime["compute_timestep_mask"],
        "debug": bool(debug),
        "debug_every": int(debug_every),
        "step_counter": 0,
    }
    model_patcher.set_attachments(_LYCORIS_TLORA_CONFIG_ATTACHMENT, config)
    model_patcher.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        _LYCORIS_TLORA_PREDICT_WRAPPER_KEY,
    )
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        _LYCORIS_TLORA_PREDICT_WRAPPER_KEY,
        _lycoris_tlora_predict_noise_wrapper,
    )


class TLoraLoaderBypass:
    """
    Load official T-LoRA attn processor checkpoints in bypass mode and
    apply timestep-dependent rank masking each denoise step.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "max_rank": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "0 = infer from checkpoint",
                    },
                ),
                "min_rank": ("INT", {"default": 1, "min": 0, "max": 4096}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
                "max_timestep": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200000,
                        "tooltip": "0 = infer from model sampling",
                    },
                ),
                "debug": ("BOOLEAN", {"default": False}),
                "debug_every": ("INT", {"default": 1, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads official T-LoRA attn-processor checkpoints and applies per-step rank masks."
    EXPERIMENTAL = True

    def _load_lora_file(self, lora_name):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            return self.loaded_lora[1]

        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_lora = (lora_path, lora)
        return lora

    def load_lora(
        self,
        model,
        clip,
        lora_name,
        strength_model,
        strength_clip,
        max_rank,
        min_rank,
        alpha,
        max_timestep,
        debug,
        debug_every,
    ):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        state_dict = self._load_lora_file(lora_name)
        model_out = model
        inferred_rank = 0

        if strength_model != 0:
            model_out, loaded_count, inferred_rank = _load_tlora_bypass_for_model(
                model=model,
                state_dict=state_dict,
                strength_model=strength_model,
            )
            logging.info("[ComfyUI-T-LoRA] Loaded %d T-LoRA adapters.", loaded_count)

            runtime_max_rank = int(max_rank) if int(max_rank) > 0 else int(inferred_rank)
            if runtime_max_rank <= 0:
                raise ValueError("Could not infer T-LoRA rank from checkpoint; set max_rank manually.")
            min_rank = _clamp_int(int(min_rank), 0, runtime_max_rank)
            _configure_tlora_runtime(
                model_patcher=model_out,
                max_rank=runtime_max_rank,
                min_rank=min_rank,
                alpha=alpha,
                max_timestep=max_timestep,
                debug=debug,
                debug_every=debug_every,
            )

        if strength_clip != 0:
            logging.warning(
                "[ComfyUI-T-LoRA] strength_clip is ignored for official T-LoRA checkpoints."
            )

        return (model_out, clip)


class TLoraLoaderBypassModelOnly(TLoraLoaderBypass):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "max_rank": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "0 = infer from checkpoint",
                    },
                ),
                "min_rank": ("INT", {"default": 1, "min": 0, "max": 4096}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
                "max_timestep": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200000,
                        "tooltip": "0 = infer from model sampling",
                    },
                ),
                "debug": ("BOOLEAN", {"default": False}),
                "debug_every": ("INT", {"default": 1, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(
        self,
        model,
        lora_name,
        strength_model,
        max_rank,
        min_rank,
        alpha,
        max_timestep,
        debug,
        debug_every,
    ):
        model_out, _ = self.load_lora(
            model=model,
            clip=None,
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=0.0,
            max_rank=max_rank,
            min_rank=min_rank,
            alpha=alpha,
            max_timestep=max_timestep,
            debug=debug,
            debug_every=debug_every,
        )
        return (model_out,)


class LycorisTLoraLoaderBypass(TLoraLoaderBypass):
    """
    Load LyCORIS T-LoRA checkpoints and apply timestep-dependent rank masking
    using LyCORIS's native mask API each denoise step.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "max_rank": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "0 = infer from checkpoint",
                    },
                ),
                "min_rank": ("INT", {"default": 1, "min": 0, "max": 4096}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
                "max_timestep": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200000,
                        "tooltip": "0 = infer from model sampling",
                    },
                ),
                "mask_group_id": ("INT", {"default": 0, "min": 0, "max": 32}),
                "debug": ("BOOLEAN", {"default": False}),
                "debug_every": ("INT", {"default": 1, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lycoris_tlora"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads LyCORIS T-LoRA checkpoints and applies per-step rank masks via LyCORIS runtime APIs."
    EXPERIMENTAL = True

    def load_lycoris_tlora(
        self,
        model,
        clip,
        lora_name,
        strength_model,
        strength_clip,
        max_rank,
        min_rank,
        alpha,
        max_timestep,
        mask_group_id,
        debug,
        debug_every,
    ):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        state_dict = self._load_lora_file(lora_name)
        lycoris_runtime = _import_lycoris_runtime()
        model_out = model
        inferred_rank = 0

        if strength_model != 0:
            model_out, loaded_count, inferred_rank = _load_lycoris_tlora_bypass_for_model(
                model=model,
                state_dict=state_dict,
                strength_model=strength_model,
                mask_group_id=mask_group_id,
                lycoris_runtime=lycoris_runtime,
            )
            logging.info(
                "[ComfyUI-T-LoRA] Loaded %d LyCORIS T-LoRA adapters.",
                loaded_count,
            )

            runtime_max_rank = int(max_rank) if int(max_rank) > 0 else int(inferred_rank)
            if runtime_max_rank <= 0:
                raise ValueError(
                    "Could not infer LyCORIS T-LoRA rank from checkpoint; set max_rank manually."
                )
            min_rank = _clamp_int(int(min_rank), 0, runtime_max_rank)
            _configure_lycoris_tlora_runtime(
                model_patcher=model_out,
                lycoris_runtime=lycoris_runtime,
                max_rank=runtime_max_rank,
                min_rank=min_rank,
                alpha=alpha,
                max_timestep=max_timestep,
                mask_group_id=mask_group_id,
                debug=debug,
                debug_every=debug_every,
            )

        if strength_clip != 0:
            logging.warning(
                "[ComfyUI-T-LoRA] strength_clip is currently ignored for LyCORIS T-LoRA checkpoints."
            )

        return (model_out, clip)


class LycorisTLoraLoaderBypassModelOnly(LycorisTLoraLoaderBypass):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "max_rank": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "0 = infer from checkpoint",
                    },
                ),
                "min_rank": ("INT", {"default": 1, "min": 0, "max": 4096}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
                "max_timestep": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200000,
                        "tooltip": "0 = infer from model sampling",
                    },
                ),
                "mask_group_id": ("INT", {"default": 0, "min": 0, "max": 32}),
                "debug": ("BOOLEAN", {"default": False}),
                "debug_every": ("INT", {"default": 1, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lycoris_tlora_model_only"

    def load_lycoris_tlora_model_only(
        self,
        model,
        lora_name,
        strength_model,
        max_rank,
        min_rank,
        alpha,
        max_timestep,
        mask_group_id,
        debug,
        debug_every,
    ):
        model_out, _ = self.load_lycoris_tlora(
            model=model,
            clip=None,
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=0.0,
            max_rank=max_rank,
            min_rank=min_rank,
            alpha=alpha,
            max_timestep=max_timestep,
            mask_group_id=mask_group_id,
            debug=debug,
            debug_every=debug_every,
        )
        return (model_out,)


NODE_CLASS_MAPPINGS = {
    "TLoraLoaderBypass": TLoraLoaderBypass,
    "TLoraLoaderBypassModelOnly": TLoraLoaderBypassModelOnly,
    "LycorisTLoraLoaderBypass": LycorisTLoraLoaderBypass,
    "LycorisTLoraLoaderBypassModelOnly": LycorisTLoraLoaderBypassModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TLoraLoaderBypass": "Load T-LoRA (Bypass)",
    "TLoraLoaderBypassModelOnly": "Load T-LoRA (Bypass, Model Only)",
    "LycorisTLoraLoaderBypass": "Load T-LoRA (LyCORIS, Bypass)",
    "LycorisTLoraLoaderBypassModelOnly": "Load T-LoRA (LyCORIS, Bypass, Model Only)",
}
