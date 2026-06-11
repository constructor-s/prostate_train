from pathlib import Path
import pandas as pd

def generate_folds_json(
    splits: list[tuple[str, Path]],
    image_col: str = "t2",
    label_col: str = "segmentation",
    id_col: str = "ID",
) -> dict:
    """Build a folds dict from a list of (split_name, csv_path) pairs.

    Args:
        splits:    List of (split_name, csv_path) pairs, e.g. [("training", Path("train.csv"))].
        image_col: CSV column name for the image path.
        label_col: CSV column name for the label/segmentation path.
        id_col:    CSV column name for the integer case ID.

    Returns:
        Dict mapping split names to lists of ``{"image": ..., "label": ..., "id": ...}`` dicts.
    """
    result: dict[str, list[dict]] = {}
    for split_name, csv_path in splits:
        df = pd.read_csv(csv_path)
        result[split_name] = [
            {"image": row[image_col], "label": row[label_col], "id": str(int(row[id_col]))}
            for _, row in df.iterrows()
        ]
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Generate a folds JSON (MONAI datalist format) from CSV files."
    )
    parser.add_argument("--training", type=Path, required=True, metavar="CSV", help="CSV for the training split.")
    parser.add_argument("--testing", type=Path, required=True, metavar="CSV", help="CSV for the testing split.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--image-col", default="t2", help="CSV column for image paths (default: t2).")
    parser.add_argument("--label-col", default="segmentation", help="CSV column for label paths (default: segmentation).")
    parser.add_argument("--id-col", default="ID", help="CSV column for case IDs (default: ID).")
    args = parser.parse_args()

    folds = generate_folds_json(
        splits=[("training", args.training), ("testing", args.testing)],
        image_col=args.image_col,
        label_col=args.label_col,
        id_col=args.id_col,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(folds, indent=4))
    total = sum(len(v) for v in folds.values())
    print(f"Wrote {total} cases to {args.output}")
