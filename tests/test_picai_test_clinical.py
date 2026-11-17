"""Exercise the cohort notebook with synthetic clinical visits and image grids."""
import contextlib
import io
import json
import re
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

SOURCE = (Path(__file__).resolve().parents[1] / 'notebooks/_picai_test_clinical.py').read_text()


class ClinicalCohortTests(unittest.TestCase):
    def run_notebook(self, visits, training_study=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            picai = root / 'data/raw/zenodo/picai'
            labels = picai / 'picai_labels'
            clinical = labels / 'clinical_information'
            masks = labels / 'anatomical_delineations/whole_gland/AI/Bosma22b'
            index_dir = picai / 'record-6624726'
            for path in (clinical, masks, index_dir):
                path.mkdir(parents=True)
            if training_study is None:
                training_study = len(visits)
                visits = [*visits, dict(patient_id=999, contacts=[(2, 0)])]
            rows, images = [], []
            training_entries = []
            poor_ids = set()
            for i, visit in enumerate(visits):
                visit = visit.copy()
                contacts = visit.pop('contacts', [])
                poor_segmentation = visit.pop('poor_segmentation', False)
                affine = visit.pop('affine', np.eye(4))
                row = dict(patient_id=i + 1, study_id=100 + i, center='RUMC',
                           mri_date='2020-01-01', prostate_volume=50, psa=5,
                           psad=np.nan, lesion_PIRADS='3', histopath_type='MRBx',
                           lesion_ISUP='0', lesion_GS='0+0', case_ISUP=0)
                row.update(visit)
                rows.append(row)
                case = f"{row['patient_id']}_{row['study_id']}"
                if poor_segmentation:
                    poor_ids.add(case)
                data = np.zeros((5, 5, 5), dtype=np.uint8)
                data[2, 2, 2] = 1
                for axis, index in contacts:
                    coordinate = [2, 2, 2]
                    coordinate[axis] = index
                    data[tuple(coordinate)] = 1
                image = nib.Nifti1Image(data, affine)
                image.header.set_xyzt_units('mm')
                nib.save(image, masks / f'{case}.nii.gz')
                nib.save(image, index_dir / f'{case}.nii.gz')
                images.append(dict(ID=case, t2=f'{case}.nii.gz'))
                if i == training_study:
                    training_entries.append(dict(image=f'zenodo/picai/record-6624726/{case}.nii.gz',
                                                 label=f'zenodo/picai/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/{case}.nii.gz',
                                                 id=f'picai_{case}'))
            pd.DataFrame(rows).to_csv(clinical / 'marksheet.csv', index=False)
            pd.DataFrame(images).to_csv(index_dir / 'images_nii.csv', index=False)
            code = SOURCE.replace("REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()",
                                  f'REPO_ROOT = Path({str(root)!r})')
            code = re.sub(r'POOR_BOSMA22B_IDS = \{.*?\}',
                          f'POOR_BOSMA22B_IDS = {poor_ids!r}', code, flags=re.S)
            (root / 'configs').mkdir()
            (root / 'configs/rumcnotprostatexnoclinical_folds.json').write_text(json.dumps({'training': training_entries}))
            namespace = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, '<cohort notebook>', 'exec'), namespace)
            manifest = json.loads((root / 'configs/picai_test_clinical_folds.json').read_text())
            for entries in manifest.values():
                for entry in entries:
                    self.assertTrue((root / 'data/raw' / entry['image']).is_file())
                    self.assertTrue((root / 'data/raw' / entry['label']).is_file())
            self.assertEqual(manifest['training'], training_entries)
            return namespace['audit'].loc[namespace['audit'].patient_id.ne(999)], manifest

    def test_completeness_and_visit_ranking(self):
        audit, folds = self.run_notebook([
            dict(patient_id=1, psa=np.nan, mri_date='2018-01-01'),
            dict(patient_id=1, mri_date='2020-01-01'),
            dict(patient_id=1, mri_date='2019-01-01'),
            dict(patient_id=1, mri_date='2019-01-01'),
            dict(patient_id=2, histopath_type='RP'),
            dict(patient_id=3, lesion_PIRADS='N/A, 6'),
            dict(patient_id=4, prostate_volume=0),
            dict(patient_id=5, lesion_ISUP='N/A', lesion_GS='bad'),
            dict(patient_id=6, lesion_ISUP='N/A, 0', lesion_GS='bad'),
        ])
        self.assertEqual(audit.loc[audit.selected, 'study_id'].tolist(), [102, 104, 105, 106, 107, 108])
        self.assertEqual(len(folds['testing_RUMC']), 2)  # PSAD absent; negative biopsies qualify.
        self.assertEqual(len(folds['incomplete_clinicals']), 4)
        self.assertEqual(len(folds['excluded_due_to_duplicate_patient']), 3)
        self.assertEqual(sum(map(len, folds.values())), 10)

    def test_training_patient_precedes_contact_and_completeness(self):
        audit, folds = self.run_notebook([
            dict(patient_id=1, contacts=[(2, 0)]),
            dict(patient_id=1),
            dict(patient_id=2),
        ], training_study=0)
        self.assertEqual(audit.assignment.tolist(), ['training', 'excluded_due_to_training_patient', 'testing_RUMC'])
        self.assertEqual(len(folds['training']), 1)
        self.assertEqual(len(folds['excluded_due_to_boundary_contact']), 0)
        self.assertFalse(audit.loc[audit.patient_id.eq(1), 'selected'].any())

    def test_boundary_first_and_orientations(self):
        reverse = np.diag([1., 1., -1., 1.])
        audit, folds = self.run_notebook([
            dict(patient_id=1, contacts=[(2, 0)]),
            dict(patient_id=1, psa=np.nan),
            dict(patient_id=2, contacts=[(2, -1)], affine=reverse),
            dict(patient_id=3, contacts=[(0, 0)]),
            dict(patient_id=4, contacts=[(2, 0)], affine=reverse),
        ])
        self.assertEqual(audit.boundary_contact_faces.tolist(), ['I', '', 'I', '', 'S'])
        self.assertEqual(audit.assignment.tolist(), [
            'excluded_due_to_boundary_contact', 'incomplete_clinicals',
            'excluded_due_to_boundary_contact', 'testing_RUMC', 'excluded_due_to_boundary_contact'])
        self.assertTrue(audit.loc[audit.patient_id.eq(2), 'selected_study_id'].isna().all())
        self.assertEqual(len(folds['excluded_due_to_boundary_contact']), 3)

    def test_site_partition_and_poor_segmentation_exclusion(self):
        audit, folds = self.run_notebook([
            dict(patient_id=1, center='PCNN'),
            dict(patient_id=2, center='RUMC'),
            dict(patient_id=3, center='ZGT'),
            dict(patient_id=4, center='PCNN', poor_segmentation=True,
                 contacts=[(2, 0)]),
            dict(patient_id=4, center='PCNN', mri_date='2021-01-01'),
        ])
        self.assertNotIn('testing', folds)
        self.assertEqual([len(folds[f'testing_{site}']) for site in ('PCNN', 'RUMC', 'ZGT')], [2, 1, 1])
        self.assertEqual(len(folds['excluded_due_to_poor_bosma22b_segmentation']), 1)
        self.assertEqual(audit.loc[audit.study_id.eq(103), 'assignment'].item(),
                         'excluded_due_to_poor_bosma22b_segmentation')
        self.assertEqual(audit.loc[audit.patient_id.eq(4) & audit.selected, 'study_id'].item(), 104)


if __name__ == '__main__':
    unittest.main()
