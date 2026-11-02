import argparse
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/raw"

P158 = "zenodo/prostate158/record-6481141/prostate158_train"
PM12 = "zenodo/promise12/record-8026660/promise12"
ISBI = "tcia/isbi_mr_prostate_2013"
QIN  = "tcia/qin_prostate_repeatability"
PDX  = "tcia/prostate_diagnosis"

SEED = 42

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


def generate_pooled_folds():
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


def generate_pooled2_folds():
    p158_train = pd.read_csv(DATA / P158 / "train.csv")
    p158_val = pd.read_csv(DATA / P158 / "valid.csv")

    qin = pd.read_csv(DATA / QIN / "index.csv")
    qin["patient"] = qin["ID"].str.split("_").str[0]
    patients = qin["patient"].unique()
    train_patients, val_patients = train_test_split(patients, test_size=2, random_state=SEED)
    qin_train = qin[qin["patient"].isin(train_patients)]
    qin_val = qin[qin["patient"].isin(val_patients)]

    pdx = pd.read_csv(DATA / PDX / "index.csv")
    pdx_train, pdx_val = train_test_split(pdx, test_size=3, random_state=SEED)

    pm12 = pd.read_csv(DATA / PM12 / "pooled_nonrunmc.csv")
    pm12_train, pm12_val = train_test_split(pm12, test_size=8, random_state=SEED)

    training = (
        to_entries(p158_train, P158, "t2", "t2_wholegland_reader1", "prostate158")
        + to_entries(qin_train, QIN, "t2", "t2_wholegland", "qin")
        + to_entries(pdx_train, PDX, "t2", "label_wg", "prostate_diagnosis")
        + to_entries(pm12_train, PM12, "t2", "segmentation", "promise12")
    )

    testing = (
        to_entries(p158_val, P158, "t2", "t2_wholegland_reader1", "prostate158")
        + to_entries(qin_val, QIN, "t2", "t2_wholegland", "qin")
        + to_entries(pdx_val, PDX, "t2", "label_wg", "prostate_diagnosis")
        + to_entries(pm12_val, PM12, "t2", "segmentation", "promise12")
    )

    folds = {"training": training, "testing": testing}
    out = ROOT / "configs/pooled2_folds.json"
    out.write_text(json.dumps(folds, indent=4))
    print(f"Wrote {len(training)} training and {len(testing)} testing entries to {out}")


def generate_rumcnotprostatex_folds():
    """Partition RUMC by linked PROSTATEx patient; use Bosma22b AI labels."""
    labels = DATA / 'zenodo/picai/picai_labels'
    index_path = DATA / 'zenodo/picai/record-6624726/images_nii.csv'
    marksheet = pd.read_csv(labels / 'clinical_information/marksheet.csv')
    cases = pd.read_csv(DATA / 'github/prostatex_masks/index.csv')
    images = pd.read_csv(index_path)
    mapping = json.loads((labels / 'additional_resources/ProstateX-mapping.json').read_text())
    for frame, columns in [(marksheet, ['patient_id', 'study_id']), (cases, ['ID']), (images, ['ID'])]:
        if frame[columns].isna().any().any() or frame.duplicated(columns).any():
            raise ValueError(f'Missing or duplicate keys: {columns}')
    links = pd.DataFrame([(key.split('_')[0], value) for key, value in mapping.items()],
                         columns=['ID', 'picai_key'])
    links = links.loc[~links['ID'].duplicated(keep=False)].copy()
    links[['patient_id', 'study_id']] = links['picai_key'].str.split('_', expand=True).astype('int64')
    cohort = cases.merge(links, on='ID', how='left', validate='one_to_one').merge(
        marksheet, on=['patient_id', 'study_id'], how='left', validate='many_to_one', indicator=True)
    volume = pd.to_numeric(cohort['prostate_volume'], errors='coerce')
    linked = cohort.loc[cohort['_merge'].eq('both') & volume.gt(0) & volume.lt(float('inf'))]
    if len(linked) != 190 or linked.duplicated(['patient_id', 'study_id']).any():
        raise ValueError('Expected 190 unique linked studies; inspect cohort inputs')
    rumc = marksheet.loc[marksheet['center'].eq('RUMC')].copy()
    rumc['ID'] = rumc['patient_id'].astype(str) + '_' + rumc['study_id'].astype(str)
    rumc = rumc.merge(images[['ID', 't2']], on='ID', how='left', validate='one_to_one')
    if rumc['t2'].isna().any():
        raise ValueError('RUMC studies missing from image index')
    heldout = rumc['patient_id'].isin(linked['patient_id'])
    splits = {'training': rumc.loc[~heldout], 'testing': rumc.loc[heldout]}
    train, test = splits.values()
    if set(train['ID']) & set(test['ID']) or set(train['patient_id']) & set(test['patient_id']):
        raise ValueError('Study or patient overlap between splits')
    if (len(rumc), len(train), train.patient_id.nunique(), len(test), test.patient_id.nunique()) != (800, 609, 602, 191, 190):
        raise ValueError('RUMC split counts changed; inspect inputs before generating folds')
    if not set(linked['picai_key']) <= set(test['ID']):
        raise ValueError('Linked studies absent from testing')
    folds = {}
    for split, frame in splits.items():
        entries = []
        for row in frame.sort_values(['patient_id', 'study_id']).itertuples(index=False):
            image = (index_path.parent / row.t2).resolve()
            label = labels / 'anatomical_delineations/whole_gland/AI/Bosma22b' / f'{row.ID}.nii.gz'
            for path in (image, label):
                if not path.is_file():
                    raise FileNotFoundError(path)
            entries.append({'image': image.relative_to(DATA.resolve()).as_posix(),
                            'label': label.relative_to(DATA).as_posix(), 'id': f'picai_{row.ID}'})
        folds[split] = entries
    out = ROOT / 'configs/rumcnotprostatex_folds.json'
    out.write_text(json.dumps(folds, indent=4))
    for split, frame in splits.items():
        print(f'{split}: {len(frame)} studies / {frame.patient_id.nunique()} patients')
    print(f'Wrote {out}; zero study and patient overlap')
    return folds


def main():
    parser = argparse.ArgumentParser(description="Generate pooled fold config JSON.")
    parser.add_argument("--pooling", choices=["pooled", "pooled2", "rumcnotprostatex"], required=True)
    args = parser.parse_args()

    if args.pooling == "pooled":
        generate_pooled_folds()
    elif args.pooling == "pooled2":
        generate_pooled2_folds()
    else:
        generate_rumcnotprostatex_folds()


if __name__ == "__main__":
    main()
