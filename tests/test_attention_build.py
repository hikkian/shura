import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
STUB = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
name=sys.argv[1]
args=sys.argv[2:]
with open(os.environ['SHURA_TEST_LOG'],'a') as f:f.write(json.dumps([name,*args])+'\\n')
state=Path(os.environ['SHURA_TEST_STATE'])
if name=='git' and 'apply' in args and args[-1].endswith('attention-decode.patch'):
    if '--reverse' in args:sys.exit(0 if (state/'applied').exists() else 1)
    if '--check' not in args:(state/'applied').touch()
if name=='cmake' and '-S' in args and os.environ.get('SHURA_TEST_FAIL_ONCE')=='1' and not (state/'failed').exists():
    (state/'failed').touch()
    sys.exit(1)
'''


class AttentionBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.dest = self.root / 'fork'
        (self.dest / '.git').mkdir(parents=True)
        (self.dest / 'build').mkdir()
        self.log = self.root / 'calls.jsonl'
        stub = self.root / 'stub.py'
        stub.write_text(STUB)
        bootstrap = self.root / 'bash-env.sh'
        bootstrap.write_text("\n".join(
            f'{name}() {{ command python3 "$SHURA_TEST_STUB" {name} "$@"; }}'
            for name in ('git', 'cmake', 'nvcc')) + "\n")
        self.env = {**os.environ, 'BASH_ENV': str(bootstrap), 'SHURA_TEST_STUB': str(stub),
                    'SHURA_TEST_LOG': str(self.log), 'SHURA_TEST_STATE': str(self.root)}

    def run_build(self, *args):
        shell = shutil.which('bash')
        self.assertIsNotNone(shell)
        return subprocess.run([shell, str(REPO / 'scripts/build-llama.sh'), *args, str(self.dest)],
                              env=self.env, capture_output=True, text=True, check=False)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_default_does_not_apply_and_explicitly_disables_cached_option(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertFalse(any('attention-decode.patch' in ' '.join(c) for c in calls))
        configure = [c for c in calls if c[0] == 'cmake' and '-S' in c]
        self.assertEqual(len(configure), 1)
        self.assertIn('-DGGML_CUDA_TURBO3_DECODE_GROUP8=OFF', configure[0])

    def test_opt_in_applies_once_and_enables_on_reused_source(self):
        for _ in range(2):
            result = self.run_build('--attention-decode')
            self.assertEqual(result.returncode, 0, result.stderr)
        applications = [c for c in self.calls() if c[0] == 'git' and 'apply' in c
                        and c[-1].endswith('attention-decode.patch') and '--check' not in c]
        self.assertEqual(len(applications), 1)
        for c in self.calls():
            if c[0] == 'cmake' and '-S' in c:
                self.assertIn('-DGGML_CUDA_TURBO3_DECODE_GROUP8=ON', c)

    def test_compiler_retry_preserves_opt_in(self):
        self.env['SHURA_TEST_FAIL_ONCE'] = '1'
        result = self.run_build('--attention-decode')
        self.assertEqual(result.returncode, 0, result.stderr)
        configure = [c for c in self.calls() if c[0] == 'cmake' and '-S' in c]
        self.assertEqual(len(configure), 2)
        self.assertTrue(all('-DGGML_CUDA_TURBO3_DECODE_GROUP8=ON' in c for c in configure))
        self.assertIn('-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler', configure[-1])

    def test_unknown_flag_fails_before_git_or_cmake(self):
        result = self.run_build('--not-a-feature')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())
