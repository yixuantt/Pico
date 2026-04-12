from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
    from safetensors.torch import save_file as safetensors_save_file
except ImportError:
    safetensors_load_file = None
    safetensors_save_file = None


_SUPPORTED_LORA_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
_ADAPTER_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
    "adapter_model.pth",
    "adapter_model.ckpt",
)


def _is_tensor_dict(obj: object) -> bool:
    return isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values())


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        if safetensors_load_file is None:
            raise ImportError(
                "Loading .safetensors requires `safetensors`. Install with: pip install safetensors"
            )
        return safetensors_load_file(str(path))

    loaded = torch.load(path, map_location="cpu")
    if _is_tensor_dict(loaded):
        return loaded
    if isinstance(loaded, dict) and _is_tensor_dict(loaded.get("state_dict")):
        return loaded["state_dict"]
    raise ValueError(f"Unsupported checkpoint format in {path}")


def _save_state_dict(state_dict: Dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        if safetensors_save_file is None:
            raise ImportError(
                "Saving .safetensors requires `safetensors`. Install with: pip install safetensors"
            )
        safetensors_save_file(state_dict, str(path))
        return
    torch.save(state_dict, path)


def _has_lora_keys(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(".lora_A." in key or ".lora_B." in key for key in state_dict)


def _discover_lora_files(input_dir: Path) -> List[Path]:
    adapter_candidates = []
    for file_name in _ADAPTER_NAMES:
        adapter_candidates.extend(input_dir.rglob(file_name))
    if adapter_candidates:
        return sorted(set(path.resolve() for path in adapter_candidates))

    generic_candidates = []
    for suffix in _SUPPORTED_LORA_SUFFIXES:
        generic_candidates.extend(input_dir.rglob(f"*{suffix}"))
    return sorted(set(path.resolve() for path in generic_candidates))


def _extract_lora_pairs(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[str, str]]:
    pairs = {}
    for key in state_dict:
        if not key.endswith(".weight") or ".lora_A." not in key:
            continue
        prefix = key[: key.index(".lora_A.")]
        suffix = key[key.index(".lora_A.") + len(".lora_A") :]
        b_key = f"{prefix}.lora_B{suffix}"
        if b_key in state_dict:
            pairs[prefix] = (key, b_key)
    return pairs


class PICO:
    """
    PICO: Pre-merge interference calibration in output-space
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        energy_compensation: bool = True,
        diagnostics_top_k: int = 5,
    ):
        self.space = "b"
        self.seed = seed
        self.energy_compensation = energy_compensation
        self.diagnostics_top_k = diagnostics_top_k
        self.layer_diagnostics: Dict[str, dict] = {}

    def __call__(self, delta_w: torch.Tensor, **kwargs) -> torch.Tensor:
        return delta_w

    def calibrate_layer(
        self,
        delta_ws: List[torch.Tensor],
        Bs: Optional[List[torch.Tensor]] = None,
        layer_name: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """Calibrate one layer in B-space."""
        num_tasks = len(delta_ws)
        if num_tasks <= 1:
            return delta_ws

        if Bs is None:
            raise ValueError("PICO B-space calibration requires Bs")
        B_all = torch.cat(Bs, dim=1).float()
        U, sigma_b, _ = torch.linalg.svd(B_all, full_matrices=False)
        variance_b = sigma_b ** 2
        total_var_b = variance_b.sum().clamp(min=1e-12)
        collision_b = variance_b / total_var_b
        alpha_b = 1.0 / (1.0 + (num_tasks - 1) * collision_b)
        coeffs_b = alpha_b - 1.0
        U_t = U.transpose(0, 1)

        calibrated = []
        for delta_w in delta_ws:
            delta_w_float = delta_w.float()
            proj_b = U_t @ delta_w_float
            correction_b = U @ (coeffs_b[:, None] * proj_b)
            delta_w_float = delta_w_float + correction_b

            calibrated.append(delta_w_float.to(delta_w.dtype))

        if layer_name is not None:
            diag = {
                "num_tasks": num_tasks,
                "space": self.space,
            }
            top_k_b = min(self.diagnostics_top_k, sigma_b.numel())
            b_diag = {
                "stack_shape": list(B_all.shape),
                "top_singular_values": sigma_b[:top_k_b].tolist(),
                "top_collision_degrees": collision_b[:top_k_b].tolist(),
                "top_alpha_coefficients": alpha_b[:top_k_b].tolist(),
                "max_collision_degree": collision_b.max().item() if collision_b.numel() else 0.0,
                "min_alpha": alpha_b.min().item() if alpha_b.numel() else 1.0,
                "avg_alpha": alpha_b.mean().item() if alpha_b.numel() else 1.0,
            }
            diag["b_space"] = b_diag
            diag.update(
                {
                    "B_all_shape": b_diag["stack_shape"],
                    "top_singular_values": b_diag["top_singular_values"],
                    "top_collision_degrees": b_diag["top_collision_degrees"],
                    "top_alpha_coefficients": b_diag["top_alpha_coefficients"],
                    "max_collision_degree": b_diag["max_collision_degree"],
                    "min_alpha": b_diag["min_alpha"],
                    "avg_alpha": b_diag["avg_alpha"],
                }
            )

            self.layer_diagnostics[layer_name] = diag

        return calibrated

    def compensate_energy(
        self,
        merged_delta_w: torch.Tensor,
        original_delta_ws: List[torch.Tensor],
    ) -> torch.Tensor:
        """Rescale merged update to match average source update energy."""
        if not self.energy_compensation or not original_delta_ws:
            return merged_delta_w

        target_energy = sum(delta_w.norm(p="fro").item() for delta_w in original_delta_ws) / len(
            original_delta_ws
        )
        merged_energy = merged_delta_w.norm(p="fro").item()
        if merged_energy <= 0:
            return merged_delta_w
        return merged_delta_w * (target_energy / max(merged_energy, 1e-8))

    def calibrate_lora_folder(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        strict: bool = False,
    ) -> Dict[str, int]:
        """
        Calibrate all LoRA adapters under one directory.

        Expected LoRA key format:
        - <prefix>.lora_A.weight
        - <prefix>.lora_B.weight

        Notes:
        - This folder mode calibrates B-space only.
        - For each layer and each adapter, optional energy compensation preserves
          the per-adapter layer update norm after calibration.
        """
        input_path = Path(input_dir).expanduser().resolve()
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"input_dir must be an existing directory, got {input_dir!r}")

        if output_dir is None:
            output_path = input_path.parent / f"{input_path.name}_calibrated"
        else:
            output_path = Path(output_dir).expanduser().resolve()

        files = _discover_lora_files(input_path)
        if not files:
            raise ValueError(f"No LoRA files found under {input_path}")

        adapter_records = []
        skipped_files = 0
        for path in files:
            try:
                state_dict = _load_state_dict(path)
            except Exception:
                skipped_files += 1
                continue
            if not _has_lora_keys(state_dict):
                skipped_files += 1
                continue
            pairs = _extract_lora_pairs(state_dict)
            if not pairs:
                skipped_files += 1
                continue
            adapter_records.append(
                {
                    "path": path,
                    "state_dict": state_dict,
                    "pairs": pairs,
                }
            )

        if len(adapter_records) < 2:
            raise ValueError(
                f"Need at least 2 valid LoRA adapters for calibration, found {len(adapter_records)}"
            )

        common_layers = set(adapter_records[0]["pairs"].keys())
        for record in adapter_records[1:]:
            common_layers &= set(record["pairs"].keys())

        if not common_layers:
            raise ValueError("No shared LoRA layers across adapters.")

        calibrated_layers = 0
        for layer_name in sorted(common_layers):
            As = []
            Bs = []
            for record in adapter_records:
                a_key, b_key = record["pairs"][layer_name]
                As.append(record["state_dict"][a_key].detach().clone().float())
                Bs.append(record["state_dict"][b_key].detach().clone().float())

            coeffs_b = None
            U = None
            B_all = torch.cat(Bs, dim=1)
            U, sigma_b, _ = torch.linalg.svd(B_all, full_matrices=False)
            collision_b = (sigma_b ** 2) / (sigma_b ** 2).sum().clamp(min=1e-12)
            alpha_b = 1.0 / (1.0 + (len(adapter_records) - 1) * collision_b)
            coeffs_b = alpha_b - 1.0

            for idx, record in enumerate(adapter_records):
                a_key, b_key = record["pairs"][layer_name]
                orig_A = As[idx]
                orig_B = Bs[idx]
                new_A = orig_A
                new_B = orig_B

                proj_b = U.transpose(0, 1) @ orig_B
                new_B = orig_B + U @ (coeffs_b[:, None] * proj_b)

                if self.energy_compensation:
                    old_delta = orig_B @ orig_A
                    new_delta = new_B @ new_A
                    old_norm = old_delta.norm(p="fro").item()
                    new_norm = new_delta.norm(p="fro").item()
                    if old_norm > 0 and new_norm > 0:
                        new_B = new_B * (old_norm / new_norm)

                record["state_dict"][a_key] = new_A.to(record["state_dict"][a_key].dtype)
                record["state_dict"][b_key] = new_B.to(record["state_dict"][b_key].dtype)

            calibrated_layers += 1

        written_files = 0
        for record in adapter_records:
            src = record["path"]
            rel = src.relative_to(input_path)
            dst = output_path / rel
            _save_state_dict(record["state_dict"], dst)
            written_files += 1

        if strict and skipped_files > 0:
            raise ValueError(
                f"Skipped {skipped_files} files that were not valid LoRA checkpoints. "
                "Set strict=False to allow skipping."
            )

        return {
            "num_input_files": len(files),
            "num_valid_loras": len(adapter_records),
            "num_skipped_files": skipped_files,
            "num_common_layers": len(common_layers),
            "num_calibrated_layers": calibrated_layers,
            "num_written_files": written_files,
        }


def _demo_run() -> None:
    """
    Minimal standalone demo:
    python -m pico.pico
    """
    torch.manual_seed(42)
    out_dim, in_dim, rank, num_tasks = 16, 32, 8, 3
    As = [torch.randn(rank, in_dim) for _ in range(num_tasks)]
    Bs = [torch.randn(out_dim, rank) for _ in range(num_tasks)]
    delta_ws = [B @ A for B, A in zip(Bs, As)]

    pico = PICO(energy_compensation=True, diagnostics_top_k=3)
    calibrated_delta_ws = pico.calibrate_layer(delta_ws, Bs=Bs, layer_name="demo_layer")
    merged_delta_w = sum(calibrated_delta_ws) / len(calibrated_delta_ws)
    compensated = pico.compensate_energy(merged_delta_w, original_delta_ws=delta_ws)

    print("PICO demo finished")
    print(f"space={pico.space}, energy_compensation={pico.energy_compensation}")
    print(f"input shape={tuple(delta_ws[0].shape)}, output shape={tuple(compensated.shape)}")
    print("diagnostics keys:", list(pico.layer_diagnostics.get("demo_layer", {}).keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PICO LoRA calibration")
    parser.add_argument("--input-dir", type=str, default=None, help="Folder containing LoRA checkpoints")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder to write calibrated LoRAs. Default: <input-dir>_calibrated",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--disable-energy-compensation",
        action="store_true",
        help="Disable per-layer norm preservation after calibration",
    )
    parser.add_argument("--diagnostics-top-k", type=int, default=5)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any file in folder is not a valid LoRA checkpoint",
    )
    args = parser.parse_args()

    pico = PICO(
        seed=args.seed,
        energy_compensation=not args.disable_energy_compensation,
        diagnostics_top_k=args.diagnostics_top_k,
    )

    if args.input_dir:
        stats = pico.calibrate_lora_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            strict=args.strict,
        )
        print("PICO folder calibration finished")
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        _demo_run()
