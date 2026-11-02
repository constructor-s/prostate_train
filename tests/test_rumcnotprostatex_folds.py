"""Local-data integration checks for the fixed RUMC split."""
import contextlib
import io
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / 'src/generate_pooled_folds.py'))


@unittest.skipUnless((ROOT / 'data/raw/zenodo/picai/record-6624726/images_nii.csv').exists(), 'Local PI-CAI required')
class TestRumcSplit(unittest.TestCase):
    def generate(self):
        with patch.object(Path, 'write_text') as write, contextlib.redirect_stdout(io.StringIO()):
            result = MODULE['generate_rumcnotprostatex_folds']()
        return result, write

    def test_membership_and_determinism(self):
        first, _ = self.generate()
        second, _ = self.generate()
        self.assertEqual(first, second)
        self.assertEqual([len(first[k]) for k in ('training', 'testing')], [609, 191])
        ids = {k: {r['id'] for r in rows} for k, rows in first.items()}
        patients = {k: {x.split('_')[1] for x in values} for k, values in ids.items()}
        self.assertFalse(ids['training'] & ids['testing'])
        self.assertFalse(patients['training'] & patients['testing'])
        self.assertTrue({'picai_11383_1001406', 'picai_11383_1001407'} <= ids['testing'])
        audit = pd.read_csv(ROOT / 'data/raw/zenodo/picai/picai_labels/clinical_information/marksheet.csv')
        expected = {'picai_' + str(r.patient_id) + '_' + str(r.study_id)
                    for r in audit.loc[audit.center.eq('RUMC')].itertuples()}
        self.assertEqual(ids['training'] | ids['testing'], expected)

    def test_missing_files_do_not_write(self):
        with patch.object(Path, 'is_file', return_value=False), patch.object(Path, 'write_text') as write:
            with self.assertRaises(FileNotFoundError):
                MODULE['generate_rumcnotprostatex_folds']()
            write.assert_not_called()

    def test_duplicate_or_missing_keys_do_not_write(self):
        read_csv = pd.read_csv
        for missing in (False, True):
            def invalid(path, *args, **kwargs):
                frame = read_csv(path, *args, **kwargs)
                if Path(path).name == 'marksheet.csv':
                    if missing:
                        frame.loc[0, 'patient_id'] = float('nan')
                    else:
                        frame = pd.concat([frame, frame.iloc[:1]])
                return frame
            with patch.object(pd, 'read_csv', side_effect=invalid), patch.object(Path, 'write_text') as write:
                with self.assertRaises(ValueError):
                    MODULE['generate_rumcnotprostatex_folds']()
                write.assert_not_called()
