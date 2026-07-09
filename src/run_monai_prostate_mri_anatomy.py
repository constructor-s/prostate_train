"""
Download and run the MONAI prostate_mri_anatomy bundle on T2w MRI images.

Usage
-----
Download the bundle (only needed once):
    python run_anatomy_inference.py download

Run inference on the Prostate158 validation split:
    python run_anatomy_inference.py infer --split valid

Run inference on the training split:
    python run_anatomy_inference.py infer --split train

Override the bundle or output directory:
    python run_anatomy_inference.py infer --bundle_dir /my/bundle --output_dir /my/results

Run inference on the ProstateX dataset:
    python src/run_monai_prostate_mri_anatomy.py infer --dataset_dir="data/raw/github/prostatex_masks" --split="index" --output_dir="results/prostatex_pred_monai_prostate"
"""

import shutil
import sys
from pathlib import Path

import fire
from monai.bundle import download as monai_bundle_download, run as monai_bundle_run

BUNDLE_NAME = "prostate_mri_anatomy"
HF_REPO = "MONAI/prostate_mri_anatomy"
HF_COMMIT = "c88463972f1be72a14817e8decdf1192b7885317"

DEFAULT_BUNDLE_DIR = Path.home() / ".cache" / "torch" / "hub" / "bundle" / BUNDLE_NAME
DEFAULT_DATASET_DIR = Path("data/raw/zenodo/prostate158/record-6481141/prostate158_train")
DEFAULT_OUTPUT_DIR = Path("results/anatomy")


def download():
    """Download the bundle from HuggingFace at the pinned commit."""
    monai_bundle_download(
        name=BUNDLE_NAME,
        source="huggingface_hub",
        repo=HF_REPO,
        version=HF_COMMIT,
    )


def infer(
    bundle_dir: str = str(DEFAULT_BUNDLE_DIR),
    dataset_dir: str = str(DEFAULT_DATASET_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    split: str = "valid",
):
    """
    Run anatomy segmentation using the MONAI bundle runner.

    The bundle expects dataset_dir to contain a test.csv with a 't2' column.
    --split names the CSV stem to use (e.g. "valid" or "train"); a temporary
    test.csv is copied from it and removed after inference.

    Outputs are written to output_dir as NIfTI files in the bundle's native
    format (one sub-folder per case, named after the input file).
    """
    bundle_dir = Path(bundle_dir)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    if not (bundle_dir / "models" / "model.pt").exists():
        sys.exit(
            f"Model not found at {bundle_dir / 'models' / 'model.pt'}.\n"
            "Run `python run_anatomy_inference.py download` first."
        )

    src_csv = dataset_dir / f"{split}.csv"
    if not src_csv.exists():
        sys.exit(f"{src_csv} not found")

    test_csv = dataset_dir / "test.csv"
    created_temp = src_csv != test_csv
    if created_temp:
        shutil.copy2(src_csv, test_csv)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference ({split}.csv): {dataset_dir} → {output_dir}")
    try:
        monai_bundle_run(
            run_id="run",
            config_file=str(bundle_dir / "configs" / "inference.json"),
            bundle_root=str(bundle_dir),
            dataset_dir=str(dataset_dir) + "/",
            output_dir=str(output_dir),
            # Preserve per-case subdirs (train/020/, train/021/, …) in the output.
            # Without this, every case writes to the same t2/t2_trans.nii.gz.
            **{"postprocessing#transforms#3#data_root_dir": str(dataset_dir)},
        )
    finally:
        if created_temp:
            test_csv.unlink(missing_ok=True)
    print("Done.")


if __name__ == "__main__":
    fire.Fire({"download": download, "infer": infer})
