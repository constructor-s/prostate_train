"""Train a direct prostate volume regression baseline from raw nnUNet data.

Purpose:
    Train a MONAI/PyTorch 3D CNN that predicts whole-gland prostate volume
    directly from raw T2-weighted MRI volumes. Label masks are used only to
    compute target volume; they are never model inputs.

Inputs:
    Raw nnUNet dataset plus the corresponding preprocessed plans/fold file:
    - ``--raw-root`` containing ``Dataset274_raw/datalist.json``.
    - ``--preprocessed-root`` containing ``Dataset274_raw/splits_final.json``
      and ``nnUNetPlans.json``.

Architecture:
    Default is ``monai.networks.nets.resnet18`` with ``spatial_dims=3``, one
    input channel, and one scalar output. ``--model resnet34`` is available as a
    capacity check. The scalar output is trained against ``log_volume_ml`` using
    ``torch.nn.SmoothL1Loss`` and AdamW.

Preprocessing and augmentation assumptions:
    Inputs are loaded directly from raw NIfTI volumes and centered or padded to
    the ``3d_fullres`` ``patch_size`` in ``nnUNetPlans.json`` by default.
    ``--spatial-size`` overrides this in nnUNet tensor order ``D H W``. Augmentations
    are intentionally regression-safe: flips, small translations, noise, and
    contrast shifts. Random zoom/scale is not used because physical size is the
    prediction target.

Outputs:
    ``--output-dir`` receives ``best.pt``, ``last.pt``, ``history.csv``,
    ``val_predictions.csv``, ``test_predictions.csv``, and ``metrics.json``.
    Predictions include the direct model output and a training-set
    median-volume baseline.

Smoke example:
    uv run python src/volume_regression/train.py \
      --raw-root work_dir_pooled2/nnunet_raw \
      --preprocessed-root work_dir_pooled2/nnunet_preprocessed \
      --output-dir /tmp/volume_regression_smoke \
      --epochs 1 --batch-size 1 --num-workers 0 --device cpu \
      --spatial-size 32 64 64 --limit-train 2 --limit-val 2 --no-augment

Full baseline example:
    uv run python src/volume_regression/train.py \
      --raw-root work_dir_pooled2/nnunet_raw \
      --preprocessed-root work_dir_pooled2/nnunet_preprocessed \
      --output-dir work_dir_volume_regression_resnet/resnet18_pooled2 \
      --model resnet18 --epochs 100 --lr 1e-4 --weight-decay 1e-4 \
      --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import (
    center_crop_or_pad,
    compute_mask_volume_ml,
    find_dataset_dir,
    load_image_volume,
    load_nnunet_raw_cases,
    load_splits,
    load_nnunet_patch_size,
    json_safe,
    regression_metrics,
    resolve_path,
)


class VolumeRegressionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        spatial_size: tuple[int, int, int],
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.spatial_size = spatial_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        row = self.frame.iloc[index]
        arr = load_image_volume(row["image"])
        arr = center_crop_or_pad(arr, self.spatial_size)
        arr = self._augment(arr) if self.augment else arr
        arr = np.expand_dims(arr, axis=0)
        return {
            "image": torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)),
            "target": torch.tensor([float(row["log_volume_ml"])], dtype=torch.float32),
            "volume_ml": float(row["volume_ml"]),
            "case_id": str(row["case_id"]),
            "dataset": str(row.get("dataset", "")),
            "split": str(row.get("split", "")),
        }

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=np.float32, copy=True)
        for axis in range(out.ndim):
            if random.random() < 0.5:
                out = np.flip(out, axis=axis)
        if random.random() < 0.5:
            shifts = [random.randint(-4, 4), random.randint(-4, 4), random.randint(-1, 1)]
            out = np.roll(out, shift=shifts, axis=(0, 1, 2))
        if random.random() < 0.3:
            out = out + np.random.normal(0.0, 0.03, size=out.shape).astype(np.float32)
        if random.random() < 0.3:
            out = (out - out.mean()) * random.uniform(0.85, 1.15) + out.mean()
        return np.ascontiguousarray(out)


def make_model(name: str) -> torch.nn.Module:
    from monai.networks.nets import resnet18, resnet34

    if name == "resnet18":
        return resnet18(spatial_dims=3, n_input_channels=1, num_classes=1)
    if name == "resnet34":
        return resnet34(spatial_dims=3, n_input_channels=1, num_classes=1)
    raise ValueError(f"Unsupported model: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    median_volume_ml: float,
) -> tuple[list[dict], dict[str, float]]:
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            pred_log = model(image).detach().cpu().numpy().reshape(-1)
            true_ml = np.asarray(batch["volume_ml"], dtype=float)
            pred_ml = np.exp(pred_log)
            for i, case_id in enumerate(batch["case_id"]):
                rows.append(
                    {
                        "case_id": case_id,
                        "split": batch["split"][i],
                        "dataset": batch["dataset"][i],
                        "volume_ml_true": float(true_ml[i]),
                        "volume_ml_pred": float(pred_ml[i]),
                        "log_volume_true": math.log(float(true_ml[i])),
                        "log_volume_pred": float(pred_log[i]),
                        "abs_error_ml": float(abs(pred_ml[i] - true_ml[i])),
                        "percent_error": float(abs(pred_ml[i] - true_ml[i]) / true_ml[i] * 100.0),
                        "median_volume_ml_pred": float(median_volume_ml),
                    }
                )
    metrics = regression_metrics(
        [r["volume_ml_true"] for r in rows],
        [r["volume_ml_pred"] for r in rows],
    )
    return rows, metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def resolve_spatial_size(args: argparse.Namespace, preprocessed_dataset_dir: Path) -> tuple[int, int, int]:
    if args.spatial_size is not None:
        return tuple(int(v) for v in args.spatial_size)

    plans_path = resolve_path(args.plans) if args.plans else preprocessed_dataset_dir / "nnUNetPlans.json"
    return load_nnunet_patch_size(plans_path, args.nnunet_config)


def compute_targets(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["volume_ml"] = rows["label"].map(compute_mask_volume_ml)
    rows["log_volume_ml"] = np.log(rows["volume_ml"])
    return rows


def build_split_frames(
    raw_root: str | Path,
    preprocessed_root: str | Path,
    dataset_id: str | int | None = 274,
    fold: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dataset_dir = find_dataset_dir(resolve_path(raw_root), dataset_id)
    preprocessed_dataset_dir = find_dataset_dir(resolve_path(preprocessed_root), dataset_id)

    raw_cases = load_nnunet_raw_cases(raw_dataset_dir)
    raw_training_cases = raw_cases[raw_cases["split"] == "training"].copy().reset_index(drop=True)
    raw_testing_cases = raw_cases[raw_cases["split"] == "testing"].copy().reset_index(drop=True)

    fold_cases = load_splits(preprocessed_dataset_dir, fold)
    fold_case_ids = set(fold_cases)
    raw_training_ids = set(raw_training_cases["case_id"])
    if fold_case_ids != raw_training_ids:
        missing_from_fold = sorted(raw_training_ids - fold_case_ids)
        extra_in_fold = sorted(fold_case_ids - raw_training_ids)
        raise ValueError(
            f"Fold {fold} does not match raw training cases for {raw_dataset_dir}: "
            f"missing_from_fold={missing_from_fold[:5]}, extra_in_fold={extra_in_fold[:5]}"
        )

    split_frame = raw_training_cases.assign(fold_split=lambda df: df["case_id"].map(fold_cases))
    train_df = split_frame[split_frame["fold_split"] == "training"].drop(columns=["fold_split"]).copy()
    val_df = split_frame[split_frame["fold_split"] == "testing"].drop(columns=["fold_split"]).copy()
    test_df = raw_testing_cases.copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Expected non-empty train/val/test splits from {raw_dataset_dir} and fold {fold}; "
            f"got train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    train_df["split"] = "training"
    val_df["split"] = "validation"
    test_df["split"] = "testing"

    return compute_targets(train_df), compute_targets(val_df), compute_targets(test_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct prostate volume regression from raw nnUNet data.")
    parser.add_argument("--raw-root", default="work_dir_pooled2/nnunet_raw")
    parser.add_argument("--preprocessed-root", default="work_dir_pooled2/nnunet_preprocessed")
    parser.add_argument("--dataset-id", default="274")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", default="results/volume_regression/resnet18_pooled2")
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--spatial-size", nargs=3, type=int, default=None)
    parser.add_argument("--plans", default=None, help="Path to nnUNetPlans.json. Defaults to the selected preprocessed dataset directory / nnUNetPlans.json.")
    parser.add_argument("--nnunet-config", default="3d_fullres")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = build_split_frames(
        raw_root=args.raw_root,
        preprocessed_root=args.preprocessed_root,
        dataset_id=args.dataset_id,
        fold=args.fold,
    )
    if args.limit_train is not None:
        train_df = train_df.head(args.limit_train)
    if args.limit_val is not None:
        val_df = val_df.head(args.limit_val)
    if args.limit_test is not None:
        test_df = test_df.head(args.limit_test)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Expected non-empty training, validation, and testing splits")

    missing_images = [p for p in pd.concat([train_df, val_df, test_df])["image"] if not Path(str(p)).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing raw image volumes, first missing: {missing_images[0]}")

    spatial_size = resolve_spatial_size(args, find_dataset_dir(resolve_path(args.preprocessed_root), args.dataset_id))
    args.spatial_size = list(spatial_size)
    print(f"Using spatial size {spatial_size} (D H W) for training and validation")
    train_ds = VolumeRegressionDataset(train_df, spatial_size, augment=not args.no_augment)
    val_ds = VolumeRegressionDataset(val_df, spatial_size, augment=False)
    test_ds = VolumeRegressionDataset(test_df, spatial_size, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = make_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss()
    median_volume_ml = float(train_df["volume_ml"].median())

    best_mae = float("inf")
    history: list[dict] = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(image)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_rows, val_metrics = predict(model, val_loader, device, median_volume_ml)
        train_loss = float(np.mean(losses))
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_mae_ml={val_metrics['mae_ml']:.3f} val_rmse_ml={val_metrics['rmse_ml']:.3f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, val_metrics, args)
        if val_metrics["mae_ml"] < best_mae:
            best_mae = val_metrics["mae_ml"]
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics, args)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    val_rows, val_metrics = predict(model, val_loader, device, median_volume_ml)
    test_rows, test_metrics = predict(model, test_loader, device, median_volume_ml)

    pd.DataFrame(val_rows).to_csv(output_dir / "val_predictions.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    metrics = {
        "direct_regression": test_metrics,
        "median_volume_baseline": regression_metrics(
            [r["volume_ml_true"] for r in test_rows],
            [r["median_volume_ml_pred"] for r in test_rows],
        ),
        "validation": {
            "direct_regression": regression_metrics(
                [r["volume_ml_true"] for r in val_rows],
                [r["volume_ml_pred"] for r in val_rows],
            ),
            "median_volume_baseline": regression_metrics(
                [r["volume_ml_true"] for r in val_rows],
                [r["median_volume_ml_pred"] for r in val_rows],
            ),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
