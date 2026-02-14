# ComfyUI-T-LoRA

Custom ComfyUI nodes for running **T-LoRA-style timestep-masked inference** on attention layers.

This repository is focused on **inference-time integration in ComfyUI**.  
For training and reference implementations, see:

- Official T-LoRA repo: https://github.com/ControlGenAI/T-LoRA

This node is currently intended on supporting only vanilla T-LoRA adapters trained using the reference code.

---

## Why This Repo Exists

Official T-LoRA pipeline logic sets a timestep-dependent rank mask before each denoise forward pass.

ComfyUI has different model execution and patch caching semantics, so this repo adapts that behavior to ComfyUI’s patcher/wrapper system.

---

## Scope

- Loads T-LoRA attention-processor checkpoints (SDXL-style `to_q/to_k/to_v/to_out` LoRA weights)
- Injects adapter math in **bypass mode** (forward-time, not static merged weights)
- Applies per-step rank mask using T-LoRA schedule
- Exposes debug logs to verify masking behavior

---

## Install

Symlink (or copy) this repo into ComfyUI custom nodes:

```bash
ln -s ~/src/ComfyUI-T-LoRA $COMFYUI_PATH/custom_nodes/ComfyUI-T-LoRA
```

Or, download it in directly:

```bash
cd $COMFYUI_PATH/custom_nodes
git clone https://github.com/bghira/ComfyUI-T-LoRA
```

Restart ComfyUI after updates.

---

## Required Model Placement

For SDXL UNet + CLIP split loading in ComfyUI:

- Base UNet checkpoint in `models/unet` (ComfyUI `diffusion_models` path)
- T-LoRA weights in `models/loras`
- CLIP text encoders in `models/text_encoders`

---

## Node Usage

Primary nodes:

- `Load T-LoRA (Bypass)` (`TLoraLoaderBypass`)
- `Load T-LoRA (Bypass, Model Only)` (`TLoraLoaderBypassModelOnly`)

Typical SDXL graph:

1. `UNETLoader` -> base SDXL UNet
2. `DualCLIPLoader` (`type=sdxl`) -> CLIP-L + CLIP-G
3. `TLoraLoaderBypass` -> apply T-LoRA weights + masking
4. Feed resulting `MODEL`/`CLIP` to sampler path

Recommended initial params:

- `strength_model = 1.0`
- `max_rank = 0` (auto-infer from checkpoint)
- `min_rank = 1`
- `alpha = 1.0`
- `max_timestep = 0` (auto)

---

## Debug Mask Verification

Set in `TLoraLoaderBypass`:

- `debug = true`
- `debug_every = 1`

You should see logs like:

- `[ComfyUI-T-LoRA][step] ... active_rank=...`
- `[ComfyUI-T-LoRA][adapter] ... active=...`

This confirms the mask is being computed and consumed during forward passes.

---

## Checkpoint Format Expectations

The loader expects official-style T-LoRA attention processor keys, e.g.:

- `...attn1.processor.to_q_lora.down.weight`
- `...attn1.processor.to_q_lora.up.weight`
- `...attn2.processor.to_out_lora.down.weight`
- `...attn2.processor.to_out_lora.up.weight`

Orthogonal variants are supported when corresponding `q_layer/p_layer/lambda/base_*` tensors exist.

<details>
<summary>Technical Note: Mapping Strategy</summary>

Keys are parsed into per-target module groups and mapped to ComfyUI UNet parameter paths using ComfyUI’s own LoRA key map utilities.  
Adapters are then installed via bypass injection hooks so adapter math runs at module forward.

</details>

---

## License / Attribution

This repo builds on ideas and checkpoint formats from the projects linked above.  
Please follow upstream licenses and citation requirements when distributing derivatives or publishing results.

