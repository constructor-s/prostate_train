from pathlib import Path

def convert_to_nifti(src_dir: Path, dst_dir: Path, pattern: str = "*.mhd") -> None:
    """Convert all PROMISE12 MHD files to NIfTI, writing to <split>_nii/ directories.

    Segmentation masks are cast to uint8 (labels are 0/1/2). Images are written
    as-is (int16). Conversion is skipped for a split if the output directory
    already contains the expected number of .nii.gz files.
    """
    import SimpleITK as sitk
    from tqdm import tqdm

    assert src_dir.is_dir(), f"Source directory does not exist: {src_dir}"
    if dst_dir.is_dir():
        assert not any(dst_dir.iterdir()), f"Destination directory already exists and is not empty: {dst_dir}"
    dst_dir.mkdir(exist_ok=True)
        
    source_files = list(src_dir.rglob(pattern))

    for mhd in tqdm(source_files, desc=f"Converting {pattern} files"):
        img = sitk.ReadImage(str(mhd))
        if "_segmentation" in mhd.name:
            img = sitk.Cast(img, sitk.sitkUInt8)
        rel_dir = mhd.relative_to(src_dir).parent
        out = dst_dir / rel_dir / (mhd.stem + ".nii.gz")
        out.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(out))
    print(f"Converted {len(source_files)} files to {dst_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="Source directory containing PROMISE12 MHD files")
    parser.add_argument("dst", type=str, help="Destination directory for NIfTI files")
    parser.add_argument("--pattern", type=str, default="*.mhd", help="File pattern to match")
    args = parser.parse_args()
    convert_to_nifti(Path(args.src), Path(args.dst), args.pattern)
