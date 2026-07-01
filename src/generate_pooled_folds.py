import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/raw"

P158 = "zenodo/prostate158/record-6481141/prostate158_train"
PM12 = "zenodo/promise12/record-8026660/promise12"
ISBI = "tcia/isbi_mr_prostate_2013"
QIN  = "tcia/qin_prostate_repeatability"


# Cases where image/label have mismatched spatial dimensions in the source data
EXCLUDE_IDS = {"isbi2013_ProstateDx-01-0055"}


def to_entries(df, prefix, img_col, lbl_col, id_prefix):
    entries = []
    for _, row in df.iterrows():
        uid = f"{id_prefix}_{row['ID']}"
        if uid in EXCLUDE_IDS:
            continue
        entries.append({
            "image": f"{prefix}/{row[img_col]}",
            "label": f"{prefix}/{row[lbl_col]}",
            "id": uid,
        })
    return entries


def main():
    qin = pd.read_csv(DATA / QIN / "index.csv")

    training = (
        to_entries(pd.read_csv(DATA / P158 / "train.csv"), P158, "t2", "t2_wholegland_reader1", "prostate158")
        + to_entries(pd.read_csv(DATA / PM12 / "training_data.csv"), PM12, "t2", "segmentation", "promise12")
        + to_entries(pd.read_csv(DATA / ISBI / "training.csv"), ISBI, "t2", "label_wg", "isbi2013")
        + to_entries(qin.iloc[0:26], QIN, "t2", "t2_wholegland", "qin")
    )

    testing = (
        to_entries(pd.read_csv(DATA / P158 / "valid.csv"), P158, "t2", "t2_wholegland_reader1", "prostate158")
        + to_entries(pd.read_csv(DATA / PM12 / "test_data.csv"), PM12, "t2", "segmentation", "promise12")
        + to_entries(pd.read_csv(DATA / ISBI / "test.csv"), ISBI, "t2", "label_wg", "isbi2013")
        + to_entries(qin.iloc[26:30], QIN, "t2", "t2_wholegland", "qin")
    )

    folds = {"training": training, "testing": testing}
    out = ROOT / "configs/pooled_folds.json"
    out.write_text(json.dumps(folds, indent=4))
    print(f"Wrote {len(training)} training and {len(testing)} testing entries to {out}")


if __name__ == "__main__":
    main()
