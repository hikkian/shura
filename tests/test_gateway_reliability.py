"""Fault injection for opt-in guard; fixtures are tiny, isolated RAM files."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock, patch

import test_gateway as fixtures

REPO = fixtures.REPO


class GuardReliability(fixtures.GatewayGuard):
    # Avoid rerunning inherited acceptance cases in this supplementary class.
    def test_tool_pauses_never_count_as_ninety_second_idle(self):
        for pause in (5, 10, 20, 89.9):
            self.gw.st.last_request = time.time() - pause
            self.tick(100)
            self.tick(110)
        self.proc.terminate.assert_not_called()
        self.assertTrue(self.gw.st.vram_pressure)

    def test_exact_ninety_second_idle_boundary(self):
        self.gw.st.last_request = 1000
        self.gw.monitor_tick(100, 100, 24, wall_now=1080)
        self.gw.monitor_tick(105, 100, 24, wall_now=1089.999)
        self.proc.terminate.assert_not_called()
        self.gw.monitor_tick(106, 100, 24, wall_now=1090)
        self.proc.terminate.assert_called_once()

    def test_emergency_waits_for_finish_even_after_ten_seconds(self):
        self.gw.st.last_request = time.time()
        self.gw.st.busy = 1
        self.tick(100, 59)
        self.tick(110, 59)
        self.assertTrue(self.gw.st.vram_emergency)
        self.proc.terminate.assert_not_called()
        self.gw.st.busy = 0
        self.tick(111, 59)
        self.proc.terminate.assert_called_once()
        self.assertEqual(self.gw.st.last_yield_reason, 'vram_emergency')

    def test_emergency_recovery_cancels_pending_unload(self):
        self.gw.st.busy = 1
        self.tick(100, 59)
        self.tick(110, 59)
        self.tick(111, 300)
        self.gw.st.busy = 0
        self.tick(112, 300)
        self.proc.terminate.assert_not_called()
        self.assertFalse(self.gw.st.vram_emergency)

    def test_emergency_without_space_unloads_and_reports_lost_context(self):
        with patch.object(self.gw, 'ram_slot_space_ok', return_value=False):
            self.tick(100, 59)
            self.tick(110, 59)
        self.proc.terminate.assert_called_once()
        self.assertEqual(list(self.disk.iterdir()), [])
        self.assertEqual(list(self.ram.iterdir()), [])
        self.assertEqual(self.gw.st.event_counters['emergency_context_lost'], 1)
        self.assertTrue(self.gw.st.skip_disk_restore)

    def test_invalid_telemetry_never_causes_emergency(self):
        for invalid in (-1, float('nan'), float('inf'), 3000000):
            self.tick(100, invalid)
            self.tick(200, invalid)
        self.proc.terminate.assert_not_called()
        self.assertEqual(self.gw.st.event_counters['telemetry_failed'], 1)
        self.tick(210, 300)
        self.assertEqual(self.gw.st.event_counters['telemetry_recovered'], 1)

    def test_reader_handles_exit_timeout_and_nonsense(self):
        # Guard telemetry is read through NVML, not a spawned nvidia-smi process.
        with patch.object(self.gw, 'nvml_device', return_value=Mock(card_free_mib=Mock(return_value=1536))):
            self.assertEqual(self.real_vram_free_gb(), 1.5)
        with patch.object(self.gw, 'nvml_device', side_effect=RuntimeError('NVML unavailable')):
            self.assertEqual(self.real_vram_free_gb(), -1)

    def test_hash_corruption_never_reaches_restore_api(self):
        self.yield_model()
        (self.ram / 'active.bin').write_bytes(b'bad checkpoint!')
        self.gw.st.model_id = self.gw.M['defaultModel']
        self.api.reset_mock()
        self.assertEqual(self.gw.restore_slot_locked(), 'restart_empty')
        self.api.assert_not_called()
        self.assertIsNone(self.gw.st.ram_checkpoint)
        self.assertEqual(list(self.ram.iterdir()), [])

    def test_wrong_restore_token_count_requires_clean_process(self):
        self.yield_model()
        self.gw.st.model_id = self.gw.M['defaultModel']
        self.api.side_effect = lambda *a, **k: (200, {'n_restored': 1})
        self.assertEqual(self.gw.restore_slot_locked(), 'restart_empty')
        self.assertIsNone(self.gw.st.ram_checkpoint)

    def test_save_atomic_rename_failure_keeps_model_and_cleans_ram(self):
        with patch.object(self.gw.os, 'replace', side_effect=OSError('injected ENOSPC')):
            self.yield_model()
        self.proc.terminate.assert_not_called()
        self.assertEqual(list(self.ram.iterdir()), [])
        self.assertIn('ENOSPC', self.gw.st.guard_fallback)

    def test_idle_save_failure_is_rate_limited(self):
        self.api.side_effect = lambda *a, **k: (500, {})
        self.yield_model()
        self.api.reset_mock()
        for now in (16, 20, 30, 44):
            self.tick(now)
        self.api.assert_not_called()
        self.tick(45)
        self.assertEqual(self.api.call_count, 1)

    def test_stop_timeout_preserves_live_pid_and_reports_failure(self):
        self.proc.terminate.side_effect = None
        self.proc.wait.side_effect = subprocess.TimeoutExpired('owned server', 10)
        self.assertFalse(self.gw.stop_llama_locked('injected test', save=False))
        self.assertIs(self.gw.st.proc, self.proc)
        self.assertEqual(self.gw.st.status, 'READY')
        self.assertEqual(self.gw.st.event_counters['stop_failed'], 1)

    def test_crash_is_observed_without_leaking_pid(self):
        self.proc.poll.return_value = -9
        self.proc.returncode = -9
        self.tick(100, 300)
        self.assertEqual(self.gw.st.status, 'UNLOADED')
        self.assertIsNone(self.gw.st.proc)
        self.assertEqual(self.gw.st.crash_count, 1)
        self.assertEqual(self.gw.st.event_counters['server_crashed'], 1)

    def test_monitor_exception_is_reported_and_next_tick_runs(self):
        stop = Mock()
        stop.wait.side_effect = [False, False, True]
        with patch.object(self.gw, 'monitor_tick', side_effect=[RuntimeError('injected'), None]) as tick:
            self.gw.monitor(stop)
        self.assertEqual(tick.call_count, 2)
        self.assertEqual(self.gw.st.event_counters['monitor_failed'], 1)
        self.assertIn('injected', self.gw.st.monitor_error)

    def test_monitor_tick_recovers_health_after_an_exception(self):
        self.gw.st.monitor_healthy = False
        self.gw.st.monitor_error = 'injected'
        self.tick(100, 300)
        self.assertTrue(self.gw.status_snapshot()['monitor_healthy'])
        self.assertEqual(self.gw.st.monitor_error, '')

    def test_checkpoint_pending_after_gateway_crash_is_cleaned(self):
        self.gw.RAM_SLOT_DIR = None
        self.gw.G['vramPressureSlotDir'] = str(self.shm_root)
        root = self.gw.prepare_ram_slot_dir()
        for name in ('active.pending.bin', 'checkpoint.json.tmp'):
            (root / name).write_bytes(b'partial')
        self.gw.RAM_SLOT_DIR = None
        self.gw.prepare_ram_slot_dir()
        self.assertEqual(list(root.iterdir()), [])

    def lease(self, pid=1234):
        value = {'pid': pid, 'start': '123', 'uid': os.getuid(), 'cmd': 'known', 'slot_dir': str(self.ram)}
        (self.ram / 'server.json').write_text(json.dumps(value))
        return {k: value[k] for k in ('pid', 'start', 'uid', 'cmd')}

    def test_reused_orphan_pid_is_never_signalled(self):
        identity = self.lease()
        with patch.object(self.gw, 'process_identity', return_value={**identity, 'start': '456'}), patch.object(self.gw.os, 'kill') as kill:
            self.gw.recover_orphan(self.ram)
        kill.assert_not_called()
        self.assertFalse((self.ram / 'server.json').exists())
        self.assertEqual(self.gw.st.event_counters['orphan_pid_reused'], 1)

    def test_busy_orphan_is_never_interrupted(self):
        identity = self.lease()
        with (patch.object(self.gw, 'process_identity', return_value=identity),
              patch.object(self.gw.os, 'kill') as kill,
              patch.object(self.gw, 'llama_call', return_value=(200, [{'is_processing': True}])),
              self.assertRaisesRegex(RuntimeError, 'busy')):
            self.gw.recover_orphan(self.ram)
        kill.assert_not_called()
        self.assertTrue((self.ram / 'server.json').exists())

    def test_absent_orphan_lease_is_removed(self):
        self.lease()
        with patch.object(self.gw, 'process_identity', side_effect=FileNotFoundError), patch.object(self.gw.os, 'kill') as kill:
            self.gw.recover_orphan(self.ram)
        kill.assert_not_called()
        self.assertFalse((self.ram / 'server.json').exists())

    def test_orphan_recovery_checks_same_pid_again_before_signal(self):
        identity = self.lease()
        with (patch.object(self.gw, 'process_identity', side_effect=[identity, {**identity, 'start': 'changed'}]),
              patch.object(self.gw.os, 'kill') as kill,
              patch.object(self.gw, 'llama_call', return_value=(200, [{'is_processing': False}])),
              self.assertRaisesRegex(RuntimeError, 'identity changed')):
            self.gw.recover_orphan(self.ram)
        kill.assert_not_called()

    def test_idle_owned_child_is_recovered_by_exact_pid(self):
        child = subprocess.Popen(['sleep', '30'], stdin=subprocess.DEVNULL)
        def cleanup():
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
        self.addCleanup(cleanup)
        for _ in range(20):
            if (Path('/proc') / str(child.pid) / 'cmdline').read_bytes().startswith(b'sleep\0'):
                break
            time.sleep(0.01)
        identity = self.gw.process_identity(child.pid)
        (self.ram / 'server.json').write_text(json.dumps({**identity, 'slot_dir': str(self.ram)}))
        with patch.object(self.gw, 'llama_call', return_value=(200, [{'is_processing': False}])):
            self.gw.recover_orphan(self.ram)
        child.wait(timeout=2)
        self.assertEqual(child.returncode, -signal.SIGTERM)
        self.assertFalse((self.ram / 'server.json').exists())

    def test_occupied_backend_port_never_spawns_a_second_server(self):
        self.gw.st.proc = None
        self.gw.st.status = 'UNLOADED'
        self.gw.backend_port_in_use.return_value = True
        with patch.object(self.gw.subprocess, 'Popen') as spawn:
            self.assertIn('occupied', self.gw.acquire_model(self.gw.M['defaultModel'], False))
        spawn.assert_not_called()
        self.assertEqual(self.gw.st.busy, 0)

    def test_ram_cleanup_failure_is_visible_and_prevents_more_copies(self):
        self.yield_model()
        with patch.object(Path, 'unlink', side_effect=OSError('injected readonly tmpfs')):
            self.assertFalse(self.gw.clear_ram_checkpoint())
        self.assertTrue(self.gw.st.ram_io_failed)
        self.assertIn('cleanup failed', self.gw.st.guard_fallback)
        self.gw.st.model_id = self.gw.M['defaultModel']
        self.api.reset_mock()
        self.assertFalse(self.gw.save_slot_locked(ram_only=True))
        self.assertEqual(self.api.call_count, 1)  # Only inspection, no additional save.

    def test_prepare_rollback_does_not_interrupt_a_response(self):
        handler = object.__new__(self.gw.Handler)
        handler.send_json = Mock()
        self.gw.st.busy = 1
        handler.guardian_endpoint('/guardian/prepare-rollback')
        self.assertEqual(handler.send_json.call_args.args[0], 409)
        self.proc.terminate.assert_not_called()
        self.assertEqual(self.gw.st.override, 'AUTO')

    def test_prepare_rollback_gates_requests_and_stops_only_idle_model(self):
        self.gw.STATE_DIR.mkdir(parents=True, exist_ok=True)
        with patch.object(self.gw, 'OVERRIDE_FLAG', self.root / 'override.flag'):
            handler = object.__new__(self.gw.Handler)
            handler.send_json = Mock()
            handler.guardian_endpoint('/guardian/prepare-rollback')
        self.assertEqual(handler.send_json.call_args.args, (200, {'ok': True, 'previous_override': 'AUTO'}))
        self.assertEqual(self.gw.st.override, 'OFF')
        self.proc.terminate.assert_called_once()
        self.assertEqual((self.root / 'override.flag').read_text(), 'OFF')

    def test_guard_switch_off_calls_preserved_entrypoint(self):
        self.gw.G.update(vramGuard=False, slotSaveCheckpoints=0)
        with patch.object(self.gw, 'legacy_main') as legacy, patch.object(self.gw, 'monitor') as monitor:
            self.gw.main()
        legacy.assert_called_once()
        monitor.assert_not_called()

    def assert_new_entrypoint(self, guard, checkpoints):
        self.gw.G.update(vramGuard=guard, slotSaveCheckpoints=checkpoints)
        self.gw.st.proc = None
        self.gw.st.ram_backend = False
        with (patch.object(self.gw, 'legacy_main') as legacy,
              patch.object(self.gw, 'ThreadingHTTPServer') as server,
              patch.object(self.gw.threading, 'Thread') as thread,
              patch.object(self.gw.signal, 'signal'),
              patch.object(self.gw, 'prepare_ram_slot_dir', side_effect=AssertionError('unexpected RAM directory'))):
            self.gw.main()
        legacy.assert_not_called()
        server.return_value.serve_forever.assert_called_once()
        thread.return_value.start.assert_called_once()
        self.assertIs(self.gw.status_snapshot()['vram_guard_enabled'], guard)
        argv = self.gw.build_args(self.gw.M['models'][self.gw.M['defaultModel']], False)
        self.assertEqual('--slot-save-checkpoints' in argv, checkpoints > 0)
        if checkpoints:
            self.assertEqual(argv[argv.index('--slot-save-checkpoints') + 1], str(checkpoints))
        if not guard:
            self.assertEqual(self.gw.server_slot_dir(), self.disk)
            self.assertFalse(self.gw.st.ram_backend)
            self.gw.st.last_request = time.time()
            self.gw.monitor_tick(1, 0, 24)
            self.assertFalse(self.gw.st.vram_pressure)
            self.assertFalse(self.gw.st.vram_emergency)
            self.assertEqual(list(self.ram.iterdir()), [])

    def test_checkpoint_only_selects_new_without_guard(self):
        self.assert_new_entrypoint(False, 1)

    def test_guard_only_selects_new(self):
        self.assert_new_entrypoint(True, 0)

    def test_guard_and_checkpoints_select_new(self):
        self.assert_new_entrypoint(True, 1)

    def test_disabled_flag_is_the_shipped_default(self):
        example = json.loads((REPO / 'config' / 'guardian.example.json').read_text())
        self.assertIs(example['vramGuard'], False)
        self.assertEqual(example['slotSaveCheckpoints'], 0)
        self.assertEqual(example['vramPressureSlotReserveMiB'], 512)

    def test_deep_slot_fits_measured_available_ram_with_reserve(self):
        self.gw.G['vramPressureSlotReserveMiB'] = 512
        self.gw.mem_available_gb.return_value = 3.5
        self.assertTrue(self.gw.ram_slot_space_ok(1_340_621_580))
        self.gw.G['vramPressureSlotReserveMiB'] = 1024
        self.assertFalse(self.gw.ram_slot_space_ok(1_340_621_580))

    def test_checkpoint_count_is_opt_in_in_test_gateway_argv(self):
        with patch.object(self.gw, 'server_slot_dir', return_value=self.disk):
            self.gw.G['slotSaveCheckpoints'] = 0
            default = self.gw.build_args(self.gw.M['models'][self.gw.M['defaultModel']], False)
            self.assertNotIn('--slot-save-checkpoints', default)
            self.gw.G['slotSaveCheckpoints'] = 1
            enabled = self.gw.build_args(self.gw.M['models'][self.gw.M['defaultModel']], False)
        self.assertEqual(enabled[enabled.index('--slot-save-checkpoints') + 1], '1')

    def test_chat_template_kwargs_reach_the_server_only_when_configured(self):
        mc = dict(self.gw.M['models'][self.gw.M['defaultModel']])
        with patch.object(self.gw, 'server_slot_dir', return_value=self.disk):
            mc.pop('chatTemplateKwargs', None)
            self.assertNotIn('--chat-template-kwargs', self.gw.build_args(mc, False))
            mc['chatTemplateKwargs'] = {'terse': False}
            argv = self.gw.build_args(mc, False)
        self.assertEqual(json.loads(argv[argv.index('--chat-template-kwargs') + 1]), {'terse': False})

    def test_authors_sampling_options_reach_the_server_only_when_configured(self):
        mc = dict(self.gw.M['models'][self.gw.M['defaultModel']])
        with patch.object(self.gw, 'server_slot_dir', return_value=self.disk):
            for key in ('minP', 'presencePenalty', 'repeatPenalty'):
                mc.pop(key, None)
            argv = self.gw.build_args(mc, False)
            for flag in ('--min-p', '--presence-penalty', '--repeat-penalty'):
                self.assertNotIn(flag, argv)
            mc.update({'temperature': 1.0, 'presencePenalty': 1.5, 'minP': 0.0})
            argv = self.gw.build_args(mc, False)
        self.assertEqual(argv[argv.index('--presence-penalty') + 1], '1.5')
        self.assertEqual(argv[argv.index('--min-p') + 1], '0.0')
        self.assertEqual(argv[argv.index('--temp') + 1], '1.0')

    def test_a_model_switch_counts_the_running_models_memory_as_available(self):
        gw = self.gw
        mc = dict(gw.M['models'][gw.M['defaultModel']])
        gw.G['ramFreeMinGBToLoad'] = 14.0
        gw.mem_available_gb.return_value = 7.7                  # the loaded model holds ~11 GB, the desktop the rest
        gw.st.vram_sample_at = time.time()
        gw.st.vram_free_mib = 325
        gw.st.monitor_error = ''
        gw.st.override = 'AUTO'
        gw.st.multimedia_lock = False
        mc['vramFreeMinGBToLoad'] = 10.0
        self.assertIn('Not enough free RAM', gw.preload_check(mc))                         # without the credit a switch is refused
        self.assertEqual(gw.preload_check(mc, reclaim=(11.3, 10.3)), '')                   # with it, the switch is admitted
        self.assertIn('Not enough free', gw.preload_check(mc, reclaim=(11.3, 1.0)))        # but not when the card really lacks memory
        class Dead:
            def poll(self):
                return 1
        gw.st.proc = Dead()
        self.assertEqual(gw.running_footprint(), (0.0, 0.0))                              # a stopped server frees nothing
        gw.st.proc = None

    def test_image_cap_and_extra_args_reach_the_server_and_vision_overrides_apply(self):
        base = dict(self.gw.M['models'][self.gw.M['defaultModel']])
        for key in ('imageMaxTokens', 'extraArgs'):
            base.pop(key, None)
        with patch.object(self.gw, 'server_slot_dir', return_value=self.disk):
            plain = self.gw.build_args(base, False)
            self.assertNotIn('--image-max-tokens', plain)
            self.assertNotIn('-ot', plain)
            mc = {**base, 'imageMaxTokens': 2048, 'extraArgs': ['-ot', r'blk\.40\.ffn_(up|down|gate)_exps\.weight=CPU']}
            argv = self.gw.build_args(mc, False)
            self.assertEqual(argv[argv.index('--image-max-tokens') + 1], '2048')
            self.assertEqual(argv[argv.index('-ot') + 1], r'blk\.40\.ffn_(up|down|gate)_exps\.weight=CPU')
            with self.assertRaises(ValueError):
                self.gw.build_args({**base, 'extraArgs': 'not a list'}, False)

    def test_terse_switch_is_put_into_chat_requests_and_wins_over_the_config(self):
        gw = self.gw
        body = json.dumps({'model': 'x', 'messages': [], 'chat_template_kwargs': {'other': 1}}).encode()
        gw.st.terse = None
        self.assertEqual(gw.apply_terse(body, '/v1/chat/completions'), body)          # no switch: the request is untouched
        gw.st.terse = False
        sent = json.loads(gw.apply_terse(body, '/v1/chat/completions'))
        self.assertEqual(sent['chat_template_kwargs'], {'other': 1, 'terse': False})
        self.assertEqual(gw.apply_terse(body, '/v1/models'), body)                    # only chat requests
        self.assertEqual(gw.apply_terse(b'not json', '/v1/chat/completions'), b'not json')
        self.assertIs(gw.terse_effective(), False)
        gw.st.terse = None
        mc = gw.M['models'][gw.M['defaultModel']]
        old = mc.get('chatTemplateKwargs')
        try:
            mc['chatTemplateKwargs'] = {'terse': False}
            self.assertIs(gw.terse_effective(), False)
            mc.pop('chatTemplateKwargs')
            self.assertIs(gw.terse_effective(), True)                                 # the template's own default
        finally:
            if old is not None:
                mc['chatTemplateKwargs'] = old

    def test_terse_switch_is_only_for_models_whose_template_has_it(self):
        gw = self.gw
        body = json.dumps({'model': 'occamy', 'messages': []}).encode()
        models = gw.M['models']
        base = dict(models[gw.M['defaultModel']])
        models['tiel-x'] = {**base, 'chatTemplateKwargs': {'terse': False}}
        models['occamy-x'] = {k: v for k, v in base.items() if k != 'chatTemplateKwargs'}
        try:
            self.assertTrue(gw.supports_terse('tiel-x'))
            self.assertFalse(gw.supports_terse('occamy-x'))
            gw.st.terse = True
            self.assertEqual(json.loads(gw.apply_terse(body, '/v1/chat/completions', 'tiel-x'))['chat_template_kwargs'], {'terse': True})
            self.assertEqual(gw.apply_terse(body, '/v1/chat/completions', 'occamy-x'), body)
            models['occamy-x']['terseSwitch'] = True                      # an explicit config entry wins
            self.assertTrue(gw.supports_terse('occamy-x'))
            snap = gw.status_snapshot()
            self.assertIn('tiel-x', snap['terse_models'])
        finally:
            gw.st.terse = None
            models.pop('tiel-x', None)
            models.pop('occamy-x', None)

    def test_terse_flag_survives_a_restart(self):
        flag = Path(self.disk) / 'terse.flag'
        with patch.object(self.gw, 'TERSE_FLAG', flag):
            self.assertIsNone(self.gw.read_terse_flag())
            flag.write_text('OFF')
            self.assertIs(self.gw.read_terse_flag(), False)
            flag.write_text('ON')
            self.assertIs(self.gw.read_terse_flag(), True)

    def test_preserved_gateway_matches_repository_baseline(self):
        # _gateway_legacy.py is the gateway as it was before the VRAM guard (commit d1f967e), kept for rollback.
        # Pinned by digest: it must never change by accident (comparing with HEAD would only hold before the guard commit).
        self.assertEqual(hashlib.sha256((REPO / 'gateway' / '_gateway_legacy.py').read_bytes()).hexdigest(),
                         '0b3a2b65bc884a46f0e147cec3233719f69c687ea98e9a6d83c1b193e0524b0b')


def load_tests(loader, standard_tests, pattern):
    import unittest
    names = [name for name in GuardReliability.__dict__ if name.startswith('test_')]
    return unittest.TestSuite(GuardReliability(name) for name in sorted(names))
