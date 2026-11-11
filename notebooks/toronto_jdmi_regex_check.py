# %% [markdown]
# # Toronto JDMI PSA report regex check
#
# Check whether each eligible case's `psa_report_ref` appears after a PSA label
# in the `areportllm` report text. Matching is case-insensitive and treats the
# PSA value as a literal string, so punctuation in the value has no regex meaning.
# The cohort is the raw full clinical listing used by
# `toronto_jdmi_ellipsoid_tolerance.py`, after the shared accession exclusions.
# Audits are written to `results/toronto_jdmi_regex_check/`.

# %%
from getpass import getpass
from io import BytesIO
import json
from pathlib import Path
import re

import msoffcrypto
import pandas as pd
from IPython.display import display

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
assert (REPO_ROOT / 'pyproject.toml').is_file(), 'Set REPO_ROOT to the repository root'
DATA_ROOT = REPO_ROOT / 'data_internal/hlab20260928'
CLINICAL_CSV = DATA_ROOT / 'hlab20260928_t2_image_listing_full_clinical.csv'
WORKBOOK = DATA_ROOT / 'PROSATELLMVOL20260703-12-WorkingBillShi.xlsx'
EXCLUDE_JSON = DATA_ROOT / 'exclude.json'
OUTPUT_DIR = REPO_ROOT / 'results/toronto_jdmi_regex_check'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Cohort
#
# Match the ellipsoid tolerance analysis: preserve raw listing order and remove
# every accession in the shared exclusion JSON. Validate identifiers before
# linking to the workbook.

# %%
listing = pd.read_csv(CLINICAL_CSV, dtype={'ID': 'string'})
assert {'ID', 'vol_report_ref'} <= set(listing.columns), (
    'Clinical listing requires ID and vol_report_ref'
)
parts = listing.ID.str.extract(r'^AIPR_([^_]+)_([^_]+)_([^_]+)$')
listing['amrn'] = parts[0]
listing['exam_date'] = parts[1]
listing['aano'] = parts[2]
listing['invalid_id'] = parts.isna().any(axis=1) | listing.ID.str.contains(r'[/\\]', na=True)
listing['duplicate_scan_id'] = listing.ID.duplicated(keep=False)
listing['duplicate_amrn'] = listing.amrn.duplicated(keep=False)

exclusions = json.loads(EXCLUDE_JSON.read_text())
assert isinstance(exclusions, list) and all(
    isinstance(entry, dict) and isinstance(entry.get('aano'), str) and entry['aano'].strip()
    for entry in exclusions
), 'Exclusion JSON requires a list of entries with nonempty aano strings'
excluded_aanos = {entry['aano'].strip() for entry in exclusions}
listing['aano_excluded'] = listing.aano.isin(excluded_aanos)
listing.to_csv(OUTPUT_DIR / 'cohort_audit.csv', index=False)
assert not listing[['invalid_id', 'duplicate_scan_id', 'duplicate_amrn']].any().any(), (
    'Invalid or repeated scan/patient IDs; inspect cohort_audit.csv'
)
cases = listing.loc[~listing.aano_excluded].copy().reset_index(drop=True)
cases['reported_volume_ml'] = pd.to_numeric(cases.vol_report_ref, errors='coerce')
assert cases.reported_volume_ml.gt(0).all() and cases.reported_volume_ml.notna().all(), (
    'Eligible cases require positive finite reported volumes, as in the ellipsoid notebook'
)

cohort_flow = pd.DataFrame([
    {'stage': 'Raw full clinical listing', 'n': len(listing)},
    {'stage': 'After global accession exclusions', 'n': len(cases)},
])
cohort_flow['excluded_since_previous'] = cohort_flow.n.shift() - cohort_flow.n
cohort_flow.to_csv(OUTPUT_DIR / 'cohort_flow.csv', index=False)
display(cohort_flow)

# %% [markdown]
# ## Workbook linkage and PSA text match
#
# Read `psa_report_ref` as text directly from Excel to retain its source string.
# The encrypted-workbook path decrypts in memory. A match requires `PSA`, any
# characters, then the literal PSA reference value. A missing report or value is
# unassessable and is recorded separately from a negative match.

# %%
with WORKBOOK.open('rb') as stream:
    office = msoffcrypto.OfficeFile(stream)
    try:
        if office.is_encrypted():
            password = getpass('Excel password: ')
            try:
                office.load_key(password=password, verify_password=True)
            finally:
                del password
            with BytesIO() as decrypted:
                office.decrypt(decrypted)
                decrypted.seek(0)
                workbook = pd.read_excel(
                    decrypted, engine='openpyxl', dtype={'psa_report_ref': str},
                    keep_default_na=False,
                )
        else:
            workbook = pd.read_excel(
                WORKBOOK, engine='openpyxl', dtype={'psa_report_ref': str},
                keep_default_na=False,
            )
    finally:
        del office

#%%
required = {'aano', 'amrn', 'psa_report_ref', 'areportllm'}
assert workbook.columns.is_unique and required <= set(workbook.columns), (
    'Workbook requires unique columns and aano, amrn, psa_report_ref, areportllm'
)
fields = workbook[['aano', 'amrn', 'psa_report_ref', 'areportllm']].copy()
fields['source_excel_row'] = range(2, len(fields) + 2)
fields['aano'] = fields.aano.astype('string').str.strip()
fields['amrn'] = fields.amrn.astype('string').str.strip().str.removeprefix('AIPR_')
fields['psa_report_ref'] = fields.psa_report_ref.astype('string').str.strip()
fields['psa_report_ref_match'] = fields.psa_report_ref.map(
    lambda value: format(float(value), '.15g')
    if pd.notna(value) and value.strip() and pd.notna(pd.to_numeric(value, errors='coerce'))
    else value
)
fields['areportllm'] = fields.areportllm.astype('string')
fields['invalid_aano'] = fields.aano.isna() | fields.aano.eq('')
fields['duplicate_aano'] = fields.aano.duplicated(keep=False)
fields.to_csv(OUTPUT_DIR / 'workbook_key_audit.csv', index=False)
assert not fields[['invalid_aano', 'duplicate_aano']].any().any(), (
    'Invalid or duplicate workbook accession keys; inspect workbook_key_audit.csv'
)

fields = fields.rename(columns={'amrn': 'workbook_amrn'})
audit = cases.merge(fields, on='aano', how='left', sort=False, validate='one_to_one',
                    indicator='workbook_match')
assert audit.ID.tolist() == cases.ID.tolist(), 'Workbook linkage changed listing order'
audit['amrn_matches'] = audit.amrn.eq(audit.workbook_amrn).fillna(False)
audit['psa_regex_match'] = pd.Series(pd.NA, index=audit.index, dtype='boolean')
audit['psa_regex_wildcard_length'] = pd.Series(pd.NA, index=audit.index, dtype='Int64')
audit['psa_regex_full_match'] = pd.Series(pd.NA, index=audit.index, dtype='string')
assessable = audit.psa_report_ref_match.notna() & audit.psa_report_ref_match.ne('') & audit.areportllm.notna()
for index in audit.index[assessable]:
    value = audit.at[index, 'psa_report_ref_match']
    report = audit.at[index, 'areportllm']
    number_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    }
    value_variants = [value, value.replace('.', ','), value.replace('.', '. ')]
    if value in number_words:
        value_variants.append(number_words[value])
    value_variants = dict.fromkeys(value_variants)
    matched = False
    for value_variant in value_variants:
        pattern = re.compile(
            r'(?:P\s*S\s*A|prostate\s+specific\s+antigen)'
            r'(?P<wildcard>.{0,100}?)' + re.escape(value_variant),
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(report)
        if match:
            matched = True
            audit.at[index, 'psa_regex_wildcard_length'] = len(match.group('wildcard'))
            audit.at[index, 'psa_regex_full_match'] = match.group(0)
            break
    audit.at[index, 'psa_regex_match'] = matched
audit['psa_regex_status'] = audit.psa_regex_match.map(
    {True: 'Match', False: 'No match'}
).fillna('Unassessable')
audit.to_csv(OUTPUT_DIR / 'psa_regex_audit.csv', index=False)
assert audit.workbook_match.eq('both').all() and audit.amrn_matches.all(), (
    'Workbook linkage or AMRN mismatch; inspect psa_regex_audit.csv'
)

#
# ## Review no-match cases
#
# Show only rows with a nonempty report, nonempty reference PSA, and a `No match`
# status. Print a 120-character context on either side of the reference value
# if it occurs anywhere in the report; otherwise show context around the first
# case-insensitive `PSA`. If neither text is present, show the complete report.

#
review = audit.loc[
    audit.areportllm.notna()
    & audit.areportllm.ne('')
    & audit.psa_report_ref.notna()
    & audit.psa_report_ref.ne('')
    & audit.psa_report_ref.ne('unavailable')
    & audit.psa_regex_status.eq('No match')
].copy()

CONTEXT_CHARS = 20

def report_context(report, psa_reference):
    reference_match = re.search(re.escape(psa_reference), report, flags=re.IGNORECASE)
    match = reference_match or re.search(r'PSA', report, flags=re.IGNORECASE)
    if match is None:
        return report
    start = max(0, match.start() - CONTEXT_CHARS)
    end = min(len(report), match.end() + CONTEXT_CHARS)
    return report[start:end]

review['areportllm_context'] = [
    report_context(report, psa_reference)
    for report, psa_reference in zip(review.areportllm, review.psa_report_ref_match)
]
with pd.option_context('display.max_colwidth', None):
    display(review[['ID', 'psa_report_ref', 'psa_regex_status', 'areportllm_context']])

# 333abfe654ebdbd: full report not available for review, maybe got anonymized away
# 2ec950e7ce3f116: no PSA in anonymized report, maybe got anonymized away
# 2ec9512e46eade6: CLINICAL: PSA of 34. Parsed as 4.5
# 2ec9503393e2c33: Clinical history: PSA 12, BPH. Parsed as 6

# %% [markdown]
# ## Review selected PSA values
#
# Inspect every cohort report whose normalized reference PSA is exactly 4.5 or
# 5.8, using the same context extraction as the no-match review above.

# %%
SELECTED_PSA_VALUES = {'4.5', '5.8'}
selected_psa_review = audit.loc[
    audit.areportllm.notna()
    & audit.areportllm.ne('')
    & audit.psa_report_ref_match.isin(SELECTED_PSA_VALUES)
].copy()
selected_psa_review['areportllm_context'] = [
    report_context(report, psa_reference)
    for report, psa_reference in zip(
        selected_psa_review.areportllm, selected_psa_review.psa_report_ref_match,
    )
]
with pd.option_context('display.max_colwidth', None, 'display.max_rows', None):
    display(selected_psa_review[[
        'ID', 'psa_report_ref', 'psa_regex_status', 'areportllm_context',
    ]])

# %% [markdown]
# ## Reference PSA value counts
#
# List normalized reference PSA values from most frequent to least frequent to
# help identify other repeated or potentially notable values for review.

# %%
psa_value_counts = (
    audit.loc[
        audit.psa_report_ref_match.notna()
        & audit.psa_report_ref_match.ne('')
        & audit.psa_report_ref_match.ne('unavailable'),
        'psa_report_ref_match',
    ]
    .value_counts()
    .rename_axis('psa_report_ref_normalized')
    .rename('n')
    .reset_index()
)
display(psa_value_counts)

# %% [markdown]
# ## Longest successful regex wildcard spans
#
# Rank successful PSA label-to-value matches by the number of characters
# captured by the regex wildcard, which is limited to at most three characters.

# %%
TOP_WILDCARD_MATCHES = 20
top_wildcard_matches = (
    audit.loc[audit.psa_regex_match.eq(True), [
        'ID', 'psa_report_ref', 'psa_regex_wildcard_length', 'psa_regex_full_match',
    ]]
    .sort_values('psa_regex_wildcard_length', ascending=False)
    .head(TOP_WILDCARD_MATCHES)
    .reset_index(drop=True)
)
with pd.option_context('display.max_colwidth', None):
    display(top_wildcard_matches)


