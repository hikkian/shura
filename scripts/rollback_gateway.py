#!/usr/bin/env python3
"""Operator-invoked rollback of the opt-in guard. No action unless invoked explicitly.

python3 scripts/rollback_gateway.py --config-dir config

Keeps the original gateway snapshot, backs up guardian.json, sets vramGuard=false atomically,
then restarts the existing user service. Busy responses finish first; no root privileges.
The llama-server binary is unchanged by Part A, so it needs no replacement on this rollback.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def call(url, path, body=None):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.loads(response.read())


def atomic_write(path, data):
    fd, name = tempfile.mkstemp(prefix='.guardian-rollback-', dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(fd, path.stat().st_mode & 0o777)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restart():
    subprocess.run(['systemctl', '--user', 'restart', 'ai-gateway.service'], check=True, timeout=90)


RESTART_WAIT = 60     # a gateway that restores its state on start (and may load a model) needs more than a few seconds


def wait_gateway(url, timeout=RESTART_WAIT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return call(url, '/guardian/status')
        except (OSError, ValueError):
            time.sleep(.25)
    raise RuntimeError('Gateway did not return after restart')


def rollback(config_dir):
    path = Path(config_dir) / 'guardian.json'
    if not path.is_file():
        raise RuntimeError(f'Config not found: {path} (use --config-dir to point at the folder that holds guardian.json)')
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise RuntimeError('Config must be a regular file owned by the current user')
    original = path.read_bytes()
    config = json.loads(original)
    if config.get('vramGuard') is not True:
        return {'changed': False, 'reason': 'Guard already disabled'}
    missing = [key for key in ('gatewayHost', 'gatewayPort') if key not in config]
    if missing:
        raise RuntimeError(f"{path} has no {' or '.join(missing)}: cannot tell which gateway to roll back")
    if config['gatewayHost'] not in ('127.0.0.1', 'localhost', '::1'):
        raise RuntimeError('Rollback is limited to a local gateway')
    host = '[::1]' if config['gatewayHost'] == '::1' else config['gatewayHost']
    url = f"http://{host}:{config['gatewayPort']}"
    # Create the backup before gating requests; failure cannot leave the service disabled.
    fd, name = tempfile.mkstemp(prefix='guardian.json.before-rollback-', dir=path.parent)
    backup = Path(name)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(original)
    deadline = time.monotonic() + 90
    while True:
        try:
            result = call(url, '/guardian/prepare-rollback', {})
            break
        except urllib.error.HTTPError as e:
            e.close()
            if e.code != 409 or time.monotonic() >= deadline:
                raise
            time.sleep(.5)
    previous = result['previous_override']
    try:
        config['vramGuard'] = False
        atomic_write(path, (json.dumps(config, indent=2) + '\n').encode())
        restart()
        status = wait_gateway(url)
        if status.get('vram_guard_enabled'):
            raise RuntimeError('Restart still serves the experimental gateway')
    except Exception as error:
        # Put everything back, and try every step even if an earlier one fails: /prepare-rollback persisted "AI off", and
        # the only thing that clears it is ai-on. The first error is the one the operator needs to see.
        problems = []

        def attempt(label, action):
            try:
                action()
            except Exception as failure:
                problems.append(f'{label}: {failure}')

        attempt('restoring guardian.json', lambda: atomic_write(path, original))
        attempt('restarting the gateway', restart)
        attempt('waiting for the gateway', lambda: wait_gateway(url))
        if previous != 'OFF':
            attempt('switching the AI back on', lambda: call(url, '/guardian/ai-on', {}))
        if problems:
            raise RuntimeError(f"Rollback failed ({error}) and the recovery was incomplete - {'; '.join(problems)}. "
                               f"The AI may still be switched OFF: run `shura on`.") from error
        raise
    if previous != 'OFF':
        try:
            call(url, '/guardian/ai-on', {})
        except Exception as error:
            raise RuntimeError(f'Rolled back, but switching the AI back on failed ({error}). Run `shura on`.') from error
    return {'changed': True, 'backup': str(backup), 'gateway': call(url, '/guardian/status')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'config')
    args = parser.parse_args()
    try:
        print(json.dumps(rollback(args.config_dir), indent=2))
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'rollback_gateway: {error}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
