"""Package a trained nnUNet model into a MONAI bundle.

Usage:
    python generate_bundle.py --config 3d_fullres --input_config ./configs/prostate158_input.yaml --output ./bundles
    python generate_bundle.py --config 2d         --input_config ./configs/prostate158_input.yaml --output ./bundles
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import yaml

TEMPLATE_DIR = Path(__file__).parent / "bundle_template"


def build_bundle(config: str, input_config: str, output_dir: str) -> None:
    cfg = yaml.safe_load(open(input_config))

    results_root = Path(cfg["nnunet_results"]).resolve()
    dataset_id = cfg["dataset_name_or_id"]
    # Resolve the dataset folder: accept either a full name or a numeric/string ID by
    # finding the matching Dataset<id>_* directory under nnunet_results.
    matching = [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith(f"Dataset{dataset_id}_")]
    if not matching:
        raise FileNotFoundError(f"No dataset folder matching Dataset{dataset_id}_* under {results_root}")
    nnunet_model_folder = matching[0] / f"nnUNetTrainer_100epochs__nnUNetPlans__{config}"

    bundle_root = Path(output_dir) / f"prostate158_{config}_5fold"
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "models").mkdir(exist_ok=True)
    (bundle_root / "scripts").mkdir(exist_ok=True)

    print(f"Packaging {config} — 5 folds into {bundle_root}")
    for fold in range(5):
        print(f"  fold {fold}...")
        fold_dst = bundle_root / "models" / f"fold_{fold}"
        fold_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nnunet_model_folder / f"fold_{fold}" / "checkpoint_best.pth",  fold_dst / "best_model.pt")
        # shutil.copy2(nnunet_model_folder / f"fold_{fold}" / "checkpoint_final.pth", fold_dst / "model.pt")

    shutil.copy2(nnunet_model_folder / "plans.json",   bundle_root / "models" / "plans.json")
    shutil.copy2(nnunet_model_folder / "dataset.json", bundle_root / "models" / "dataset.json")

    write_metadata(bundle_root, config, cfg)
    shutil.copy2(TEMPLATE_DIR / "infer.py",    bundle_root / "scripts" / "infer.py")
    shutil.copy2(TEMPLATE_DIR / "Dockerfile",  bundle_root / "Dockerfile")
    print(f"Done → {bundle_root}")


def write_metadata(bundle_root: Path, config: str, cfg: dict) -> None:
    inference_info_path = (
        Path(cfg["nnunet_results"]) / "Dataset158_prostate158_train" / "inference_information.json"
    )
    dice_scores = {}
    if inference_info_path.exists():
        dice_scores = json.load(open(inference_info_path)).get("all_results", {})

    metadata = {
        "name": f"prostate158_{config}_5fold",
        "description": f"Prostate158 segmentation — nnUNet {config}, 5-fold ensemble",
        "authors": ["BillS"],
        "created": str(date.today()),
        "nnunet_config": config,
        "trainer": "nnUNetTrainer_100epochs",
        "plans": "nnUNetPlans",
        "folds": [0, 1, 2, 3, 4],
        "inference_checkpoint": "best_model.pt",
        "dice_scores_crossval": dice_scores,
        "inference": {
            "description": "Run scripts/infer.py for 5-fold ensemble inference via nnUNetPredictor",
            "example": "python scripts/infer.py /path/to/bundle /input /output",
        },
    }
    json.dump(metadata, open(bundle_root / "metadata.json", "w"), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a MONAI bundle from a trained nnUNet model.")
    parser.add_argument("--config", required=True, choices=["3d_fullres", "2d"])
    parser.add_argument("--input_config", default="./configs/prostate158_input.yaml")
    parser.add_argument("--output", default="./bundles")
    args = parser.parse_args()
    build_bundle(args.config, args.input_config, args.output)


if __name__ == "__main__":
    main()
