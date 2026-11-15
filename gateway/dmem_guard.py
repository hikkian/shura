"""MEMV2 (item 6): the desktop's VRAM protected with the dmem cgroup controller, as a last insurance above the governor.

When the card's memory runs out, amdgpu evicts buffers; the dmem controller (Linux 6.14+, amdgpu 6.15+) lets the kernel prefer
the buffers of a cgroup above its protection. With dmem.low = max on the desktop's slices (session.slice: KWin and Plasma;
app.slice: the browser and the other applications) and no protection on shura.slice, the driver evicts Shura's buffers first.
KFD evicts a ROCm process as a whole, so the model then freezes until its memory comes back: an insurance, not a mechanism.

Root is needed once, for the levels systemd does not delegate: dmem has to be enabled in user.slice and user-UID.slice
(ROOT_DROPIN below, for /etc/systemd/system/user@.service.d/). Everything under user@UID.service belongs to the user: enabling
dmem there and writing dmem.low needs no root. systemd (259) does not know dmem, so nothing of this lives in unit files.

Pure functions over a cgroup root path, so the tests use a fake tree. Sizes are bytes as the kernel prints them.
"""

from pathlib import Path

DESK_SLICES = ("session.slice", "app.slice")
SHURA_SLICE = "shura.slice"
ROOT_DROPIN_PATH = "/etc/systemd/system/user@.service.d/shura-dmem.conf"
# %i of user@.service is the user's UID; "+" runs the line with full privileges although the unit runs as the user
ROOT_DROPIN = """# Shura (MEMV2): let the user manager use the dmem controller (device memory) for its slices.
# Remove this file and run `systemctl daemon-reload` to undo; the controller goes away at the next login.
[Service]
ExecStartPre=+/bin/sh -c 'echo +dmem > /sys/fs/cgroup/user.slice/cgroup.subtree_control; echo +dmem > /sys/fs/cgroup/user.slice/user-%i.slice/cgroup.subtree_control'
"""
GATEWAY_SLICE_DROPIN = """# Shura (MEMV2): the gateway and its model server in their own slice, outside the desktop's dmem protection.
[Service]
Slice=shura.slice
"""


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def regions(cgroup_root="/sys/fs/cgroup"):
    """The device-memory regions of the machine ({name: capacity bytes}) from dmem.capacity; {} without dmem."""
    text = read(Path(cgroup_root) / "dmem.capacity") or ""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
    return out


def vram_regions(cgroup_root="/sys/fs/cgroup"):
    return [name for name in regions(cgroup_root) if name.startswith("drm/") and name.endswith("/vram")]


def user_manager(uid, cgroup_root="/sys/fs/cgroup"):
    return Path(cgroup_root) / "user.slice" / f"user-{uid}.slice" / f"user@{uid}.service"


def controller_on(cgroup):
    """dmem is enabled for the children of `cgroup` (its subtree_control lists it)."""
    return "dmem" in (read(Path(cgroup) / "cgroup.subtree_control") or "").split()


def status(uid, cgroup_root="/sys/fs/cgroup"):
    """What is in place: the root part, the user part, the protection per desk slice, and where Shura's cgroup is."""
    root = Path(cgroup_root)
    manager = user_manager(uid, root)
    out = {
        "regions": vram_regions(root),
        "root_part": controller_on(root / "user.slice") and controller_on(root / "user.slice" / f"user-{uid}.slice"),
        "user_part": controller_on(manager),
        "slices": {},
    }
    for name in DESK_SLICES + (SHURA_SLICE,):
        path = manager / name
        out["slices"][name] = {"exists": path.is_dir(), "low": (read(path / "dmem.low") or "").strip(),
                               "current": (read(path / "dmem.current") or "").strip()}
    return out


def where_is(pid, proc_root="/proc"):
    """The cgroup path of a process (cgroup v2), or None."""
    text = read(Path(proc_root) / str(pid) / "cgroup") or ""
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:]
    return None


def shura_outside_desk(cgroup_path):
    """True when a cgroup path (as /proc/PID/cgroup gives it) is not under one of the protected desk slices."""
    if not cgroup_path:
        return False
    parts = cgroup_path.strip("/").split("/")
    return not any(p in DESK_SLICES for p in parts)


def plan(uid, cgroup_root="/sys/fs/cgroup"):
    """The user-level writes, as (path, text) in order, and the reasons they cannot be done yet. No root anywhere."""
    root = Path(cgroup_root)
    manager = user_manager(uid, root)
    vram = vram_regions(root)
    blockers = []
    if not vram:
        blockers.append("the kernel has no dmem VRAM region (needs Linux 6.14+ and amdgpu 6.15+)")
    if not (controller_on(root / "user.slice") and controller_on(root / "user.slice" / f"user-{uid}.slice")):
        blockers.append(f"the root part is missing: {ROOT_DROPIN_PATH}, then log out and in")
    writes = [(manager / "cgroup.subtree_control", "+dmem")]
    for name in DESK_SLICES:
        if (manager / name).is_dir():
            for region in vram:
                writes.append((manager / name / "dmem.low", f"{region} max"))
    return writes, blockers


def undo_writes(uid, cgroup_root="/sys/fs/cgroup"):
    """Writes that take the user-level part back (protection to 0); the controller itself goes with the root part."""
    root = Path(cgroup_root)
    manager = user_manager(uid, root)
    writes = []
    for name in DESK_SLICES:
        if (manager / name / "dmem.low").exists():
            for region in vram_regions(root):
                writes.append((manager / name / "dmem.low", f"{region} 0"))
    return writes


def apply(writes, writer=None):
    """Perform writes in order; returns the errors as text (an empty list when all went through)."""
    errors = []
    for path, text in writes:
        try:
            if writer is not None:
                writer(path, text)
            else:
                Path(path).write_text(text + "\n")
        except OSError as error:
            errors.append(f"{path}: {error.strerror or error}")
    return errors
