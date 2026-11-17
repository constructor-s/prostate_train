# %% [markdown]
# # Paired volume errors for the site-specific PI-CAI manuscript
#
# Reuse current strict-intersection case exports for ProstateX, Toronto manual,
# expanded Toronto, and each PI-CAI institution. No thresholds, statistics,
# predictions, or cohort memberships are recalculated. Signed error is estimated
# minus reference volume, in mL. Screening is independent within each PI-CAI site.
# Bosma22b and expanded Toronto references are automated surrogates.

# %%
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
OUTPUT_DIR = REPO_ROOT / 'results/paper_picai_sites'
SOURCE_ROOT = REPO_ROOT / 'results/volume_disagreement_quality_improvement'
SITES = ('PCNN', 'RUMC', 'ZGT')
manifest_path = REPO_ROOT / 'configs/picai_test_clinical_folds.json'
manifest = json.loads(manifest_path.read_text())
input_hashes = {str(manifest_path.relative_to(REPO_ROOT)): hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
tables = {}
for site in SITES:
    path = SOURCE_ROOT / site / 'case_audit.csv'
    metadata_path = SOURCE_ROOT / site / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    assert metadata['site'] == site and metadata['fold'] == f'testing_{site}'
    for source in metadata['cohorts']:
        for relative_path, digest in source['input_sha256'].items():
            assert hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest() == digest, relative_path
    frame = pd.read_csv(path)
    expected_ids = {entry['id'] for entry in manifest[f'testing_{site}']}
    site_frame = frame.loc[frame.cohort.eq('picai')]
    assert site_frame.ID.is_unique and set(site_frame.ID) == expected_ids
    for source in (path, metadata_path):
        input_hashes[str(source.relative_to(REPO_ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    tables[site] = frame

# %% [markdown]
# ## Verify the unchanged datasets and assemble six panels
#
# Non-PI-CAI rows are repeated by the upstream site runs. Require identical
# values before using one copy. Each PI-CAI panel uses only its own site export.

# %%
panels = []
for key, label in [('prostatex', 'ProstateX'), ('toronto_manual', 'Toronto manual'),
                   ('toronto_surrogate', 'Expanded Toronto')]:
    baseline = tables['PCNN'].loc[tables['PCNN'].cohort.eq(key)].sort_values('ID').reset_index(drop=True)
    for site in SITES[1:]:
        comparison = tables[site].loc[tables[site].cohort.eq(key)].sort_values('ID').reset_index(drop=True)
        pd.testing.assert_frame_equal(baseline, comparison)
    panels.append((label, baseline.loc[baseline.flag_both]))
for site in SITES:
    frame = tables[site]
    panels.append((f'PI-CAI {site}', frame.loc[frame.cohort.eq('picai') & frame.flag_both]))

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
for letter, ax, (label, selected) in zip('abcdef', axes.flat, panels):
    for row in selected.itertuples():
        color = '#228833' if row.absolute_error_reduction_ml > 0 else '#EE7733' if row.absolute_error_reduction_ml < 0 else '0.5'
        ax.plot([row.gt_volume_ml, row.gt_volume_ml], [row.clinical_error_ml, row.nnunet274_error_ml],
                color=color, alpha=.65, linewidth=.9)
        ax.scatter(row.gt_volume_ml, row.clinical_error_ml, marker='o', s=20, facecolors='none', edgecolors=color)
        ax.scatter(row.gt_volume_ml, row.nnunet274_error_ml, marker='^', s=22, color=color)
    ax.axhline(0, color='black', linewidth=.7)
    ax.set_xscale('log')
    if not selected.empty:
        ticks = np.geomspace(selected.gt_volume_ml.min(), selected.gt_volume_ml.max(), 3)
        ax.set_xticks(ticks, labels=[f'{value:.0f}' for value in ticks])
        ax.set_xticks([], minor=True)
    ax.set_title(f'({letter}) {label}\nStrict intersection: n = {len(selected)}', fontsize=11)
    ax.set_xlabel('Reference volume (mL; log scale)')
    ax.set_ylabel('Estimated − reference volume (mL)')
fig.legend(handles=[
    Line2D([], [], marker='o', markerfacecolor='none', color='0.3', linestyle='none', label='Clinical ellipsoid'),
    Line2D([], [], marker='^', color='0.3', linestyle='none', label='Multicenter'),
    Line2D([], [], color='#228833', label='Lower absolute error'),
    Line2D([], [], color='#EE7733', label='Higher absolute error')],
    loc='outside lower center', ncol=4, fontsize=9)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_DIR / 'paired_volume_errors.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'paired_volume_errors.pdf')
plt.show()

# %%
metadata = dict(input_sha256=input_hashes, panels=[dict(dataset=label, n=len(frame), ids=frame.ID.tolist())
    for label, frame in panels], volume_units='mL', error_sign='estimated minus reference',
    screen='strict intersection; independently estimated within each PI-CAI institution',
    source_notebook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if '__file__' in globals() else None)
(OUTPUT_DIR / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
print(pd.DataFrame([dict(dataset=label, flagged=len(frame)) for label, frame in panels]))
