"""Reversible user-systemd CPU weight with owned-unit checks and RAM recovery lease."""

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


class Scheduler:
    def __init__(
        self,
        config,
        root="/proc",
        system=os,
        runner=subprocess.run,
        state_key="gateway",
        cgroup_root="/sys/fs/cgroup",
    ):
        self.root, self.system, self.runner = Path(root), system, runner
        self.cgroup_root = Path(cgroup_root)
        self.unit = config.get("politeUnit", "ai-gateway.service")
        self.weights = config.get("politeWeights", [100, 50, 20, 5])
        if (
            not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", self.unit)
            or len(self.weights) != 4
            or any(type(x) is not int or not 1 <= x <= 10000 for x in self.weights)
        ):
            raise ValueError("invalid polite unit/weights")
        token = hashlib.sha256((self.unit + str(state_key)).encode()).hexdigest()[:16]
        self.lease_dir = Path("/dev/shm") / f"shura-cpu-{os.getuid()}-{token}"
        self.lease = self.lease_dir / "weight.json"
        self.original = None
        self.weight = None
        self.path = None
        self.last_ms = None
        self.last_checked = 0.0
        self.affinity_active = False
        self.affinity_before = {}
        self.reserved = max(0, int(config.get("politeReserveCores", 1)))

    def info(self):
        result = self.runner(
            [
                "systemctl",
                "--user",
                "--no-ask-password",
                "show",
                self.unit,
                "-p",
                "ControlGroup",
                "-p",
                "CPUWeight",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )

    def reversible(self):
        try:
            info = self.info()
            path = info.get("ControlGroup", "")
            own = (self.root / "self/cgroup").read_text().strip().split("::", 1)[1]
            if not path or not (own == path or own.startswith(path + "/")):
                return (
                    False  # Never change the live unit from a standalone test gateway.
                )
            if not self.cgroup_root.joinpath(path.lstrip("/"), "cpu.weight").exists():
                return False
            self.path = path
            self.original = info["CPUWeight"]
            return True
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
            return False

    def set_weight(self, value):
        t = time.monotonic()
        self.runner(
            [
                "systemctl",
                "--user",
                "--no-ask-password",
                "set-property",
                "--runtime",
                self.unit,
                "CPUWeight=" + str(value),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        self.last_ms = (time.monotonic() - t) * 1000

    def write_lease(self, weight):
        self.lease_dir.mkdir(mode=0o700, exist_ok=True)
        stat = self.lease_dir.lstat()
        if (
            self.lease_dir.is_symlink()
            or stat.st_uid != os.getuid()
            or stat.st_mode & 0o077
        ):
            raise PermissionError("CPU lease directory must be private")
        tmp = self.lease_dir / "weight.pending"
        tmp.write_text(
            json.dumps(
                {
                    "unit": self.unit,
                    "original": self.original,
                    "applied": weight,
                    "previous": self.weight,
                }
            )
        )
        os.replace(tmp, self.lease)

    def recover(self):
        if not self.lease.exists():
            return
        stat = self.lease_dir.lstat()
        if (
            self.lease_dir.is_symlink()
            or stat.st_uid != os.getuid()
            or stat.st_mode & 0o077
            or self.lease.is_symlink()
            or self.lease.stat().st_size > 4096
        ):
            raise PermissionError("Unsafe CPU lease")
        saved = json.loads(self.lease.read_text())
        if (
            saved["unit"] != self.unit
            or type(saved["applied"]) is not int
            or not 1 <= saved["applied"] <= 10000
        ):
            raise ValueError("Invalid CPU lease")
        if not self.reversible():
            raise PermissionError("CPU recovery unit does not own this gateway")
        info = self.info()
        previous = saved.get("previous")
        expected = {
            str(saved["applied"]),
            str(saved["original"] if previous is None else previous),
        }
        if info["CPUWeight"] not in expected:
            raise RuntimeError(
                "CPUWeight changed externally; automatic restore refused"
            )
        value = saved["original"]
        if value != "[not set]" and (
            not str(value).isdigit() or not 1 <= int(value) <= 10000
        ):
            raise ValueError("Invalid original CPUWeight")
        self.set_weight("" if value == "[not set]" else value)
        self.lease.unlink()
        self.original = None
        self.weight = None
        self.last_checked = 0.0
        self.reversible()

    def change(self, weight):
        if self.original is None:
            if not self.reversible():
                raise PermissionError(
                    "Target unit does not own this gateway or cpu controller is unavailable"
                )
        now = time.monotonic()
        if self.weight is not None and (
            weight != self.weight or now - self.last_checked >= 5
        ):
            if self.info()["CPUWeight"] != str(self.weight):
                raise RuntimeError("CPUWeight changed externally; controller disabled")
            self.last_checked = now
        if self.weight == weight:
            return
        # Lease precedes property change so SIGKILL cannot leave an undocumented override.
        self.write_lease(weight)
        try:
            self.set_weight(weight)
            if self.info()["CPUWeight"] != str(weight):
                raise RuntimeError("CPUWeight readback mismatch")
            self.weight = weight
            self.last_checked = time.monotonic()
        except BaseException:
            self.set_weight("" if self.original == "[not set]" else self.original)
            self.lease.unlink(missing_ok=True)
            self.weight = None
            raise

    def close(self):
        if self.weight is not None:
            current = self.info()["CPUWeight"]
            if current != str(self.weight):
                raise RuntimeError("External CPUWeight change; not overwritten")
            self.set_weight("" if self.original == "[not set]" else self.original)
            self.lease.unlink(missing_ok=True)
            self.weight = None

    def cpus_for_l3(self, full):
        groups = {}
        for cpu in sorted(full):
            path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id")
            package = Path(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
            )
            try:
                key = (package.read_text().strip(), path.read_text().strip())
            except OSError:
                key = ("cpu", cpu)
            groups.setdefault(key, set()).add(cpu)
        keep = max(1, len(groups) - self.reserved)
        return set().union(*list(groups.values())[:keep])

    def apply(self, pid, identity, level, full_cpus):
        from polite_model import proc_ticks

        if proc_ticks(self.root / str(pid) / "stat")[1] != identity:
            raise ProcessLookupError("Server PID reused")
        group = (self.root / str(pid) / "cgroup").read_text().strip().split("::", 1)[1]
        if not self.path or not (
            group == self.path or group.startswith(self.path + "/")
        ):
            raise PermissionError("Server is outside the owned unit")
        previous = self.weight
        self.change(self.weights[level])
        if level != 3 and not self.affinity_active:
            return  # Static CPUWeight does not touch the model's affinity.
        cpus = self.cpus_for_l3(full_cpus) if level == 3 else full_cpus
        snapshots = []
        try:
            for entry in (self.root / str(pid) / "task").iterdir():
                tid = int(entry.name)
                start = proc_ticks(entry / "stat")[1]
                snapshots.append((tid, start, self.system.sched_getaffinity(tid)))
                key = (pid, tid, start)
                if level == 3:
                    self.affinity_before.setdefault(key, snapshots[-1][2])
                self.system.sched_setaffinity(
                    tid,
                    cpus if level == 3 else self.affinity_before.get(key, full_cpus),
                )
            self.affinity_active = level == 3
            if level != 3:
                self.affinity_before.clear()
        except OSError:
            for tid, start, mask in snapshots:
                try:
                    if (
                        proc_ticks(self.root / str(pid) / "task" / str(tid) / "stat")[1]
                        == start
                    ):
                        self.system.sched_setaffinity(tid, mask)
                except FileNotFoundError:
                    pass
            if previous is not None:
                self.change(previous)
            else:
                self.close()
            raise
