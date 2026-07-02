"""5-fold ensemble inference using nnUNetPredictor.

Usage:
    python scripts/infer.py --bundle_root <path> --input <folder|file|csv|json> --output_folder <path>

CSV input must have a 't2' column with paths relative to the CSV file's directory.

JSON input must be a fold-split file with "training"/"testing" keys (e.g.
configs/pooled2_folds.json), each a list of {"image", "label", "id"} entries
with paths relative to --data_root. Use --split to choose which section to run.
"""
import os
# Set dummy nnUNet environment variable to suppress warnings about missing environment variables. 
os.environ["nnUNet_raw"] = os.devnull
os.environ["nnUNet_preprocessed"] = os.devnull
os.environ["nnUNet_results"] = os.devnull

import argparse
import csv
import json
import pathlib
import tempfile

import nibabel
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder


def _save_as_nifti(src: str, dst: str) -> None:
    """Load any image format SimpleITK supports and save as NIfTI."""
    # If suffices are the same, just make a link instead of reading and writing the file.
    if pathlib.Path(src).suffixes == pathlib.Path(dst).suffixes:
        os.link(src, dst)
        return
    
    sitk_img = sitk.ReadImage(src)
    arr = sitk.GetArrayFromImage(sitk_img).transpose(2, 1, 0)  # (z,y,x) → (x,y,z)
    spacing = sitk_img.GetSpacing()         # (x, y, z)
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    origin = sitk_img.GetOrigin()

    affine = np.eye(4)
    affine[:3, :3] = direction * np.array(spacing)
    affine[:3, 3] = origin

    nib_img = nibabel.Nifti1Image(arr, affine)
    nibabel.save(nib_img, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 5-fold ensemble inference from a MONAI/nnUNet bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--bundle_root", default=None, help="Path to the bundle directory (contains models/); defaults to the grandparent of this script")
    parser.add_argument("--input", required=True, help="Folder of *_0000.nii.gz files, a single file, a CSV with a 't2' column, or a fold-split JSON with 'training'/'testing' keys")
    parser.add_argument("--output_folder", required=True, help="Folder where predictions will be written")
    parser.add_argument("--split", choices=["training", "testing"], default=None,
                        help="Which section of a fold-split JSON to run; required when --input is a .json file")
    parser.add_argument("--data_root", default="data/raw",
                        help="Root directory that 'image' paths in a fold-split JSON are relative to")

    # nnUNetPredictor constructor — names match the underlying API exactly
    parser.add_argument("--tile_step_size", type=float, default=0.5,
                        help="Step size between tiles as a fraction of tile size; lower = faster, higher = better overlap")
    parser.add_argument("--use_gaussian", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply Gaussian weighting at tile borders to reduce boundary artefacts")
    parser.add_argument("--use_mirroring", action=argparse.BooleanOptionalAction, default=True,
                        help="Test-time augmentation via mirroring; --no-use_mirroring to roughly halve inference time")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")

    # initialize_from_trained_model_folder
    parser.add_argument("--checkpoint_name", default="checkpoint_final.pt",
                        help="Checkpoint file to load from each fold_X/ directory")

    # predict_from_files
    parser.add_argument("--save_probabilities", action=argparse.BooleanOptionalAction, default=False,
                        help="Also save softmax probability maps alongside segmentation masks")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                        help="Re-run prediction even if output file already exists")
    parser.add_argument("--num_processes_preprocessing", type=int, default=1,
                        help="Worker processes for image preprocessing")
    parser.add_argument("--num_processes_segmentation_export", type=int, default=8,
                        help="Worker processes for writing output segmentation files")

    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if args.split is not None and input_path.suffix != ".json":
        parser.error("--split is only valid when --input is a fold-split JSON file")

    _tmp_dir_ctx = None
    if input_path.suffix == ".json":
        if args.split is None:
            parser.error("--split is required when --input is a fold-split JSON file")
        data_root = pathlib.Path(args.data_root)
        with open(input_path) as f:
            fold_data = json.load(f)
        if args.split not in fold_data:
            raise KeyError(f"'{args.split}' not found in {input_path}; available: {list(fold_data)}")
        rows = fold_data[args.split]
        _tmp_dir_ctx = tempfile.TemporaryDirectory()
        tmp_dir = _tmp_dir_ctx.name
        for row in tqdm(rows, desc="Converting images to NIfTI", delay=5):
            src = str(data_root / row["image"])
            dst = os.path.join(tmp_dir, f"{row['id']}_0000.nii.gz")
            _save_as_nifti(src, dst)
        list_of_lists_or_source_folder = tmp_dir
        print(f"Converted {len(rows)} images ({args.split}) to NIfTI in temporary folder {tmp_dir}")
    elif input_path.suffix == ".csv":
        data_root = input_path.parent
        with open(input_path) as f:
            rows = list(csv.DictReader(f))
        _tmp_dir_ctx = tempfile.TemporaryDirectory()
        tmp_dir = _tmp_dir_ctx.name
        for row in tqdm(rows, desc="Converting images to NIfTI", delay=5):
            src = str(data_root / row["t2"])
            dst = os.path.join(tmp_dir, f"{row['ID']}_0000.nii.gz")
            _save_as_nifti(src, dst)
        list_of_lists_or_source_folder = tmp_dir
        print(f"Converted {len(rows)} images to NIfTI in temporary folder {tmp_dir}")
    elif input_path.is_file():
        list_of_lists_or_source_folder = [[str(input_path)]]
    else:
        list_of_lists_or_source_folder = str(input_path)

    bundle_root = pathlib.Path(args.bundle_root) if args.bundle_root else pathlib.Path(__file__).parent.parent
    models_dir = bundle_root / "models"
    if not (models_dir / "fold_0").exists():
        raise FileNotFoundError(
            f"Expected fold_0/ not found under {models_dir}. "
            "Check that bundle_root points to the bundle directory, not models/."
        )

    predictor = nnUNetPredictor(
        tile_step_size=args.tile_step_size,
        use_gaussian=args.use_gaussian,
        use_mirroring=args.use_mirroring,
        device=torch.device(args.device),
    )
    predictor.initialize_from_trained_model_folder(
        str(models_dir),
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name=args.checkpoint_name,
    )
    try:
        predictor.predict_from_files(
            list_of_lists_or_source_folder,
            args.output_folder,
            save_probabilities=args.save_probabilities,
            overwrite=args.overwrite,
            num_processes_preprocessing=args.num_processes_preprocessing,
            num_processes_segmentation_export=args.num_processes_segmentation_export,
        )

        pp_pkl = models_dir / "postprocessing.pkl"
        if pp_pkl.exists():
            print(f"Applying postprocessing from {pp_pkl} to {args.output_folder}")
            pp_fns, pp_fn_kwargs = load_pickle(str(pp_pkl))
            apply_postprocessing_to_folder(
                input_folder=args.output_folder,
                output_folder=args.output_folder,
                pp_fns=pp_fns,
                pp_fn_kwargs=pp_fn_kwargs,
                plans_file_or_dict=str(models_dir / "plans.json"),
                dataset_json_file_or_dict=str(models_dir / "dataset.json"),
                # num_processes=args.num_processes_segmentation_export,
            )
        else:
            print(f"Warning: {pp_pkl} not found, skipping postprocessing.")
    finally:
        if _tmp_dir_ctx is not None:
            _tmp_dir_ctx.cleanup()


if __name__ == "__main__":
    main()
