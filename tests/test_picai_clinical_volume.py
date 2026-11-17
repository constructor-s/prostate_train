"""Exercise retained PI-CAI paired-inference cells with synthetic patients."""
import ast
from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/picai_test_clinical_volume.py'
SOURCE = NOTEBOOK.read_text()
STATISTICS_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## Paired post hoc inference:'):
    SOURCE.index('# %% [markdown]\n# ## PSA density: calculations')
]

class PicaiClinicalVolumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def statistics(self, patient_ids):
        n = len(patient_ids)
        volumes = pd.DataFrame({
            'patient_id': patient_ids, 'gt_volume_ml': np.full(n, 50.),
            'clinical_volume_ml': 50 + np.arange(1, n + 1),
            'nnunet274_volume_ml': 50 - np.arange(1, n + 1),
        })
        tree = ast.parse(STATISTICS_SOURCE)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'N_BOOTSTRAP':
                    node.value = ast.Constant(199)
        ast.fix_missing_locations(tree)
        namespace = dict(
            np=np, pd=pd, combinations=combinations, volumes=volumes,
            SELECTED_COMPARATORS=['clinical', 'nnunet274'],
            COMPARATORS={'clinical': ('Clinical',), 'nnunet274': ('nnUNet274',)},
            OUTPUT_DIR=self.root, display=lambda value: None,
        )
        with redirect_stdout(StringIO()):
            exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)
        return namespace['pairwise_statistics'].iloc[0]

    def test_paired_inference_uses_surrogate_reference_and_unique_patients(self):
        row = self.statistics(np.arange(12))
        self.assertEqual(row.n, 12)
        self.assertEqual(row.ties, 12)
        self.assertEqual(row.rmse_improvement_ml, 0)
        self.assertEqual(row.rmse_bootstrap_p, 1)
        self.assertEqual(row.holm_family_size, 2)
        self.assertEqual(row.closer_conclusion, 'uninformative_all_ties')
        with self.assertRaisesRegex(AssertionError, 'unique patients'):
            self.statistics(np.zeros(12))


if __name__ == '__main__':
    unittest.main()
