"""Parallel execution must use the matching MPI runtime and clean up ranks."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sram_compiler.sizing.execution import execute, execution_command
from sram_compiler.sizing.qualification import Case


class ExecutionTests(unittest.TestCase):
    def test_parallel_launcher_is_selected_beside_xyce_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            xyce, launcher = root / 'Xyce', root / 'mpiexec'
            xyce.write_text('Xyce test binary')
            launcher.write_text('matching MPI launcher')
            launcher.chmod(0o755)
            with patch('subprocess.check_output', return_value='Parallel with MPI\n'):
                command, metadata = execution_command(xyce, root / 'deck.sp', 17, 4)
            self.assertEqual(command[:4], [str(launcher), '-n', '4', str(xyce)])
            self.assertEqual(metadata['mpi_ranks'], 4)
            self.assertEqual(metadata['blas_threads_per_rank'], 1)
            self.assertEqual(len(metadata['mpi_launcher_sha256']), 64)
            self.assertIn('KLU', command)
            with patch('subprocess.check_output', return_value='Serial\n'):
                with self.assertRaisesRegex(ValueError, 'MPI-enabled'):
                    execution_command(xyce, root / 'deck.sp', 17, 4)
            launcher.unlink()
            with patch('subprocess.check_output', return_value='Parallel with MPI\n'):
                with self.assertRaisesRegex(ValueError, 'Matching MPI launcher'):
                    execution_command(xyce, root / 'deck.sp', 17, 4)

    def test_rank_count_changes_case_identity(self):
        self.assertEqual(Case(8, 4).name, Case(8, 4, mpi_ranks=1).name)
        self.assertNotEqual(Case(64, 64).name, Case(64, 64, mpi_ranks=4).name)

    def test_timeout_terminates_launcher_and_child(self):
        with tempfile.TemporaryDirectory() as temp:
            child_file = Path(temp) / 'child.pid'
            program = ('import subprocess,sys,time; '
                       'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                       f'open({str(child_file)!r},"w").write(str(p.pid)); time.sleep(60)')
            with (Path(temp) / 'log').open('w') as log:
                with self.assertRaises(subprocess.TimeoutExpired):
                    execute([sys.executable, '-c', program], log, 1)
            pid = int(child_file.read_text())
            status = Path(f'/proc/{pid}/stat')
            if status.exists():
                self.assertEqual(status.read_text().split()[2], 'Z')


if __name__ == '__main__':
    unittest.main()
