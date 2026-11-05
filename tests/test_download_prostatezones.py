"""Check extraction, patient validation, and safe reruns without network access."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import download_prostatezones as downloader


class DownloadProstateZonesTests(unittest.TestCase):
    def test_download_validates_layout_and_skips_completed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)

            def fake_download(command, check):
                self.assertTrue(check)
                if command[0] == 'git':
                    self.assertEqual(command[1:4], ['clone', '--depth', '1'])
                    self.assertEqual(command[-2], downloader.COMPANION_REPO)
                    target = data_root / 'raw/zenodo/prostatezones/ProstateZones'
                    self.assertEqual(Path(command[-1]), target)
                    (target / '.git').mkdir(parents=True)
                    return
                self.assertEqual(command[1:4], ['-m', 'zenodo_get', '10718469'])
                output = Path(command[command.index('-o') + 1])
                with ZipFile(output / downloader.ARCHIVE_NAME, 'w') as archive:
                    archive.writestr('README.txt', 'Original annotations')
                    archive.writestr('summary_cspca.xlsx', 'Summary')
                    for patient in range(160):
                        archive.writestr(f'Singles/Seg-{patient:04d}.nrrd', 'mask')
                    for reader in ('R1', 'R2'):
                        for patient in range(160, 200):
                            archive.writestr(f'Duplicates/{reader}/Seg-{patient:04d}_{reader}.nrrd', 'mask')

            with patch.object(downloader.subprocess, 'run', side_effect=fake_download) as run:
                root = downloader.download_dataset(data_root, progress=False)
                self.assertEqual(root, data_root / 'raw/zenodo/prostatezones/record-10718469')
                self.assertTrue(downloader.is_complete(root))
                manifest = json.loads((root / downloader.MANIFEST_NAME).read_text())
                self.assertEqual(manifest['unique_patients'], 200)
                self.assertEqual(manifest['total_masks'], 240)
                self.assertEqual(downloader.download_dataset(data_root, progress=False), root)
                self.assertEqual(run.call_count, 2)  # Download and companion clone.

                # A completed annotation record still obtains a missing companion.
                (data_root / 'raw/zenodo/prostatezones/ProstateZones/.git').rmdir()
                self.assertEqual(downloader.download_dataset(data_root, progress=False), root)
                self.assertEqual(run.call_count, 3)
                self.assertEqual(run.call_args.args[0][0], 'git')

                self.assertEqual(downloader.download_dataset(data_root, progress=False), root)
                self.assertEqual(run.call_count, 3)  # Both resources now skipped.

            (data_root / 'raw/zenodo/prostatezones/ProstateZones/.git').rmdir()
            with patch.object(downloader.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'git')):
                with self.assertRaises(subprocess.CalledProcessError):
                    downloader.download_dataset(data_root, progress=False)
            self.assertTrue(downloader.is_complete(root))  # Clone failure preserves annotation completion.

            # Correct counts alone must not hide mismatched duplicate identities.
            mask = root / 'Duplicates/R2/Seg-0199_R2.nrrd'
            mask.rename(root / 'Duplicates/R2/Seg-0200_R2.nrrd')
            self.assertFalse(downloader.is_complete(root))
            with patch.object(downloader.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'incomplete dataset'):
                    downloader.download_dataset(data_root)
                run.assert_not_called()

    def test_existing_incomplete_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            root = downloader.dataset_root(data_root, downloader.DEFAULT_RECORD_ID)
            root.mkdir(parents=True)
            with patch.object(downloader.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'incomplete dataset'):
                    downloader.download_dataset(data_root)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
