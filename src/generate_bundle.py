"""Package a trained nnUNet model into a MONAI bundle.

Usage:
    python src/generate_bundle.py --model_dir work_dir/nnUNet_trained_models/Dataset158_prostate158_train/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres
    python src/generate_bundle.py --model_dir work_dir/nnUNet_trained_models/Dataset158_prostate158_train/nnUNetTrainer_100epochs__nnUNetPlans__2d
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent.parent / "bundle_template"


def build_bundle(model_dir: str, output_dir: str) -> None:
    nnunet_model_folder = Path(model_dir).resolve()
    if not nnunet_model_folder.is_dir():
        raise FileNotFoundError(f"model_dir not found: {nnunet_model_folder}")

    # Derive config (e.g. "3d_fullres") and dataset slug from folder structure:
    # <results_root>/Dataset<id>_<dataset_name>/nnUNetTrainer_<X>epochs__nnUNetPlans__<config>
    parts = nnunet_model_folder.name.split("__")  # ["nnUNetTrainer_100epochs", "nnUNetPlans", "3d_fullres"]
    config = parts[-1]
    trainer = parts[0]  # e.g. "nnUNetTrainer_100epochs"
    dataset_slug = nnunet_model_folder.parent.name  # e.g. "Dataset158_prostate158_train"

    bundle_name = f"{dataset_slug}_{config}_5fold"
    bundle_root = Path(output_dir) / bundle_name
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "models").mkdir(exist_ok=True)
    (bundle_root / "scripts").mkdir(exist_ok=True)

    print(f"Packaging {config} — 5 folds into {bundle_root}")
    for fold in range(5):
        print(f"  fold {fold}...")
        fold_dst = bundle_root / "models" / f"fold_{fold}"
        fold_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nnunet_model_folder / f"fold_{fold}" / "checkpoint_final.pth", fold_dst / "checkpoint_final.pt")

    shutil.copy2(nnunet_model_folder / "plans.json",   bundle_root / "models" / "plans.json")
    shutil.copy2(nnunet_model_folder / "dataset.json", bundle_root / "models" / "dataset.json")

    crossval_dir = nnunet_model_folder / "crossval_results_folds_0_1_2_3_4"
    for src in list(crossval_dir.glob("*.pkl")) + list(crossval_dir.glob("*.json")):
        shutil.copy2(src, bundle_root / "models" / src.name)
    if not (bundle_root / "models" / "postprocessing.pkl").exists():
        print(f"Warning: postprocessing.pkl not found in {crossval_dir} — postprocessing will be skipped at inference time.")

    write_metadata(bundle_root, config, trainer, dataset_slug)
    shutil.copy2(TEMPLATE_DIR / "infer.py",   bundle_root / "scripts" / "infer.py")
    shutil.copy2(TEMPLATE_DIR / "Dockerfile", bundle_root / "Dockerfile")
    print(f"Done → {bundle_root}")


def write_metadata(bundle_root: Path, config: str, trainer: str, dataset_slug: str) -> None:
    metadata = {
        "name": bundle_root.name,
        "description": f"{dataset_slug} segmentation — nnUNet {config}, 5-fold ensemble",
        "authors": ["BillS"],
        "created": str(date.today()),
        "nnunet_config": config,
        "trainer": trainer,
        "plans": "nnUNetPlans",
        "folds": [0, 1, 2, 3, 4],
        "inference_checkpoint": "checkpoint_final.pt",
        "inference": {
            "description": "Run scripts/infer.py for 5-fold ensemble inference via nnUNetPredictor",
            "example": "python scripts/infer.py /path/to/bundle /input /output",
        },
    }
    json.dump(metadata, open(bundle_root / "metadata.json", "w"), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a MONAI bundle from a trained nnUNet model.")
    parser.add_argument("--model_dir", required=True,
                        help="Path to the nnUNet trainer+config folder (contains fold_0/, plans.json, etc.)")
    parser.add_argument("--output", default="./bundles")
    args = parser.parse_args()
    build_bundle(args.model_dir, args.output)


if __name__ == "__main__":
    main()
