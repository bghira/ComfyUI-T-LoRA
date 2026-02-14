import logging
import math
import re
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
_TLORA_KEY_PATTERN = re.compile(
    r"^(?:unet\.)?(?P<base>.+?\.to_(?:q|k|v|out(?:\.0)?))_lora\.(?P<param>.+)$"
)

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


def _resolve_unet_key(base_key: str, key_map: dict[str, str]):
    candidates = [base_key]
    if ".to_out" in base_key and ".to_out.0" not in base_key:
        candidates.append(base_key.replace(".to_out", ".to_out.0"))
    if ".to_out.0" in base_key:
        candidates.append(base_key.replace(".to_out.0", ".to_out"))

    for candidate in candidates:
        mapped = key_map.get(candidate, None)
        if mapped is not None:
            return mapped
        mapped = key_map.get(f"unet.{candidate}", None)
        if mapped is not None:
            return mapped
    return None


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
            skipped.append(key)
            continue

        grouped.setdefault(model_weight_key, {})[param_name] = value

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

    for model_weight_key, params in grouped.items():
        adapter = _build_tlora_adapter(params)
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

    for key in skipped:
        logging.warning("[ComfyUI-T-LoRA] Unmapped T-LoRA key: %s", key)

    injections = manager.create_injections(model_lora.model)
    if manager.get_hook_count() <= 0:
        raise ValueError("No T-LoRA adapters were created from this checkpoint.")

    injection_key = f"{_TLORA_INJECTION_KEY_PREFIX}_{uuid.uuid4().hex}"
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


NODE_CLASS_MAPPINGS = {
    "TLoraLoaderBypass": TLoraLoaderBypass,
    "TLoraLoaderBypassModelOnly": TLoraLoaderBypassModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TLoraLoaderBypass": "Load T-LoRA (Bypass)",
    "TLoraLoaderBypassModelOnly": "Load T-LoRA (Bypass, Model Only)",
}
