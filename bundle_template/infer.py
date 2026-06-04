"""5-fold ensemble inference using nnUNetPredictor.

Usage:
    python scripts/infer.py bundle_root input_folder output_folder [options]

Input folder must contain NIfTI files named *_0000.nii.gz (nnUNet single-modality convention).
"""
import argparse
import pathlib

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 5-fold ensemble inference from a MONAI/nnUNet bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("bundle_root",    help="Path to the bundle directory (contains models/)")
    parser.add_argument("input_folder",   help="Folder of input NIfTI images (*_0000.nii.gz), or a single *_0000.nii.gz file")
    parser.add_argument("output_folder",  help="Folder where predictions will be written")

    # nnUNetPredictor constructor — names match the underlying API exactly
    parser.add_argument("--tile_step_size", type=float, default=0.5,
                        help="Step size between tiles as a fraction of tile size; lower = faster, higher = better overlap")
    parser.add_argument("--use_gaussian", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply Gaussian weighting at tile borders to reduce boundary artefacts")
    parser.add_argument("--use_mirroring", action=argparse.BooleanOptionalAction, default=True,
                        help="Test-time augmentation via mirroring; disable to halve inference time")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")

    # initialize_from_trained_model_folder
    parser.add_argument("--checkpoint_name", default="best_model.pt",
                        help="Checkpoint file to load from each fold_X/ directory")

    # predict_from_files
    parser.add_argument("--save_probabilities", action=argparse.BooleanOptionalAction, default=False,
                        help="Also save softmax probability maps alongside segmentation masks")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                        help="Re-run prediction even if output file already exists")
    parser.add_argument("--num_processes_preprocessing", type=int, default=8,
                        help="Worker processes for image preprocessing")
    parser.add_argument("--num_processes_segmentation_export", type=int, default=8,
                        help="Worker processes for writing output segmentation files")

    args = parser.parse_args()

    input_path = pathlib.Path(args.input_folder)
    if input_path.is_file():
        list_of_lists_or_source_folder = [[str(input_path)]]
    else:
        list_of_lists_or_source_folder = str(input_path)

    models_dir = pathlib.Path(args.bundle_root) / "models"
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
    predictor.predict_from_files(
        list_of_lists_or_source_folder,
        args.output_folder,
        save_probabilities=args.save_probabilities,
        overwrite=args.overwrite,
        num_processes_preprocessing=args.num_processes_preprocessing,
        num_processes_segmentation_export=args.num_processes_segmentation_export,
    )


if __name__ == "__main__":
    main()
