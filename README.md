![Pico banner](asset/img.png)

**Crowded in B-Space: Calibrating Shared Directions for LoRA Merging**  

Pico is a data-free pre-merge calibration method for LoRA adapters.  
This repository provides a practical implementation with both Python API and CLI, including batch calibration for a folder of LoRA checkpoints.

## Usages

- Calibrate all LoRA checkpoints under one directory (recursive scan).
- Preserve per-layer update energy by default (`energy_compensation=True`).
- B-space calibration.
- Export calibrated checkpoints to a separate output directory.

## Install

Run inside your conda environment:

```bash
conda activate <your-env>
pip install torch safetensors
```

## Quick Start (Folder Mode)

### 1) Prepare input folder

Put multiple LoRA checkpoints under one directory, for example:

```text
/path/to/loras/
├── task_math/adapter_model.safetensors
├── task_code/adapter_model.safetensors
└── task_finance/adapter_model.bin
```

### 2) Run calibration

```bash
conda activate <your-env>
python -m pico.pico \
  --input-dir /path/to/loras \
  --output-dir /path/to/loras_calibrated
```

If `--output-dir` is not provided, output defaults to:

```text
<input-dir>_calibrated
```

### 3) Read result

CLI prints summary stats, for example:

- `num_input_files`
- `num_valid_loras`
- `num_skipped_files`
- `num_common_layers`
- `num_calibrated_layers`
- `num_written_files`

## CLI Reference

```bash
python -m pico.pico --help
```

Main arguments:

- `--input-dir`: folder containing LoRA checkpoints.
- `--output-dir`: destination folder for calibrated checkpoints.
- `--disable-energy-compensation`: disable per-layer norm preservation.
- `--strict`: fail if invalid checkpoints exist in the folder.
- `--diagnostics-top-k`: top-k singular/collision stats to store.

Notes:

- Calibration space is `b`.

## Supported Input Formats

File discovery strategy:

1. Prefer adapter files:
   - `adapter_model.safetensors`
   - `adapter_model.bin`
   - `adapter_model.pt`
   - `adapter_model.pth`
   - `adapter_model.ckpt`
2. Fallback to generic suffix scan:
   - `.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`

Checkpoint requirements:

- State dict must contain LoRA keys with both:
  - `.lora_A.`
  - `.lora_B.`
- Shared layers across adapters are calibrated jointly.

## Python API

```python
from pico import PICO

pico = PICO(energy_compensation=True, diagnostics_top_k=5)
stats = pico.calibrate_lora_folder(
    input_dir="/path/to/loras",
    output_dir="/path/to/loras_calibrated",
    strict=False,
)
print(stats)
```

Core class methods:

- `calibrate_layer(delta_ws, Bs=None, As=None, layer_name=None)`
- `compensate_energy(merged_delta_w, original_delta_ws)`
- `calibrate_lora_folder(input_dir, output_dir=None, strict=False)`

## Merging Calibrated LoRAs

After calibration, you can merge adapters with any merge algorithm.  
Below is a minimal Task Arithmetic example (simple average of `delta_w`):

```python
import torch

# Assume these are loaded from calibrated adapters for one layer
As = [...]  # list[Tensor], each shape [r, in_dim]
Bs = [...]  # list[Tensor], each shape [out_dim, r]

# Recover per-adapter update
delta_ws = [B @ A for A, B in zip(As, Bs)]

# Task Arithmetic merge (uniform average)
merged_delta_w = torch.stack(delta_ws, dim=0).mean(dim=0)

# Apply merged_delta_w to your base model by your own pipeline
```

You can replace the averaging line with your own merge rule (e.g., weighted averaging, TIES, TSV-style solvers).

## Minimal Demo

If you run without `--input-dir`, a synthetic tensor demo is executed:

```bash
python -m pico.pico
```

## Troubleshooting

- `ImportError: safetensors`  
  Install dependency: `pip install safetensors`

- `Need at least 2 valid LoRA adapters`  
  Folder must contain at least two valid LoRA checkpoints.

- `No shared LoRA layers across adapters`  
  Input adapters do not have overlapping LoRA layer names.

- Calibration space is B-space.  
  No `--space` argument is required.


## Citation

```bibtex
@article{tang2026crowded,
  title   = {Crowded in B-Space: Calibrating Shared Directions for LoRA Merging},
  author  = {Tang, Yixuan and Yang, Yi},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2026}
}
```

## License

See `LICENSE`.
