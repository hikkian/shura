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


def wait_gateway(url, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return call(url, '/guardian/status')
        except (OSError, ValueError):
            time.sleep(.25)
    raise RuntimeError('Gateway did not return after restart')


def rollback(config_dir):
    path = Path(config_dir) / 'guardian.json'
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise RuntimeError('Config must be a regular file owned by the current user')
    original = path.read_bytes()
    config = json.loads(original)
    if config.get('vramGuard') is not True:
        return {'changed': False, 'reason': 'Guard already disabled'}
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
    except Exception:
        atomic_write(path, original)
        restart()
        wait_gateway(url)
        if previous != 'OFF':
            call(url, '/guardian/ai-on', {})
        raise
    if previous != 'OFF':
        call(url, '/guardian/ai-on', {})
    return {'changed': True, 'backup': str(backup), 'gateway': call(url, '/guardian/status')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'config')
    args = parser.parse_args()
    print(json.dumps(rollback(args.config_dir), indent=2))


if __name__ == '__main__':
    main()
