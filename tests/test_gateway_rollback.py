"""Rollback faults are exercised only on tiny copied configs in tmpfs."""
import importlib.util
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


class Rollback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('rollback_test', Path(__file__).resolve().parent.parent / 'scripts' / 'rollback_gateway.py')
        cls.rb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.rb)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir='/tmp', prefix='shura-rollback-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.file = self.root / 'guardian.json'
        self.original = json.dumps({'vramGuard': True, 'gatewayHost': '127.0.0.1', 'gatewayPort': 8081}).encode()
        self.file.write_bytes(self.original)

    def test_rolls_back_with_backup_and_preserves_manual_off(self):
        with patch.object(self.rb, 'call', side_effect=[{'previous_override': 'OFF'}, {'override': 'OFF'}]) as call, patch.object(self.rb, 'restart') as restart, patch.object(self.rb, 'wait_gateway', return_value={'status': 'UNLOADED'}):
            result = self.rb.rollback(self.root)
        self.assertFalse(json.loads(self.file.read_bytes())['vramGuard'])
        self.assertEqual(Path(result['backup']).read_bytes(), self.original)
        restart.assert_called_once()
        self.assertEqual(call.call_count, 2)

    def test_failed_restart_restores_original_config(self):
        with (patch.object(self.rb, 'call', return_value={'previous_override': 'OFF'}),
              patch.object(self.rb, 'restart', side_effect=[RuntimeError('injected'), None]) as restart,
              patch.object(self.rb, 'wait_gateway', return_value={}),
              self.assertRaisesRegex(RuntimeError, 'injected')):
            self.rb.rollback(self.root)
        self.assertEqual(self.file.read_bytes(), self.original)
        self.assertEqual(restart.call_count, 2)

    def test_disabled_feature_does_not_write_or_restart(self):
        self.file.write_text('{"vramGuard":false}')
        before = self.file.stat().st_mtime_ns
        with patch.object(self.rb, 'call') as call, patch.object(self.rb, 'restart') as restart:
            result = self.rb.rollback(self.root)
        self.assertFalse(result['changed'])
        self.assertEqual(self.file.stat().st_mtime_ns, before)
        call.assert_not_called()
        restart.assert_not_called()

    def test_busy_response_is_waited_out_before_any_config_change(self):
        busy = urllib.error.HTTPError('http://localhost', 409, 'busy', {}, None)
        with patch.object(self.rb, 'call', side_effect=[busy, {'previous_override': 'OFF'}, {}]), patch.object(self.rb.time, 'sleep') as sleep, patch.object(self.rb, 'restart'), patch.object(self.rb, 'wait_gateway', return_value={}):
            self.rb.rollback(self.root)
        sleep.assert_called_once_with(.5)

    def test_backup_failure_does_not_disable_running_gateway(self):
        with (patch.object(self.rb.tempfile, 'mkstemp', side_effect=OSError('injected ENOSPC')),
              patch.object(self.rb, 'call') as call,
              patch.object(self.rb, 'restart') as restart,
              self.assertRaisesRegex(OSError, 'ENOSPC')):
            self.rb.rollback(self.root)
        call.assert_not_called()
        restart.assert_not_called()
        self.assertEqual(self.file.read_bytes(), self.original)

    def test_symlink_config_is_rejected(self):
        target = self.root / 'source.json'
        target.write_bytes(self.original)
        self.file.unlink()
        self.file.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, 'regular'):
            self.rb.rollback(self.root)
        self.assertEqual(target.read_bytes(), self.original)
