# prostate_train

## Download Prostate158

Install the dependency and run the script:

```bash
uv pip install zenodo-get
python download_prostate158.py
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

By default, the dataset is extracted into:

```text
data/raw/zenodo/prostate158/record-6481141/prostate158_train
```

Temporary archives are kept under `data/tmp/zenodo/...` while the download runs.
That keeps raw data reproducible and makes it easy to add more datasets later.