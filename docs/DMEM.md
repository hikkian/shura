# dmem: the desktop's VRAM comes first even when the card runs out (MEMV2)

The VRAM governor keeps free VRAM above its floor by giving back the expert cache, and desk need parks the model during a game.
`dmem` is the insurance under both: when the card's memory still runs out, amdgpu evicts buffers, and the **dmem cgroup controller**
(Linux 6.14+, amdgpu 6.15+) lets the kernel take them from the cgroup that is above its protection. With protection on the desktop's
slices and none on Shura's, the driver evicts Shura's buffers, not KWin's or the browser's.

The price: KFD (ROCm) evicts a compute process as a whole, so the model **freezes** until its memory comes back. That is why this is
the last line, not the mechanism.

## What goes where

| cgroup (under `user@UID.service`) | who | `dmem.low` |
|---|---|---|
| `session.slice` | KWin, Plasma | `max` |
| `app.slice` | the browser, editors, messengers | `max` |
| `shura.slice` | Shura's gateway and its model server | none |

Protection in the dmem controller is recursive, so Shura must **not** live inside a protected slice. Today the user service
`ai-gateway.service` runs in `app.slice`; it moves to `shura.slice` with a user-level drop-in (below).

## Steps

`scripts/shura-dmem status` shows what is in place (read only).

**1. Root, once.** systemd (259) does not know dmem and does not delegate it, so it has to be enabled for `user.slice` and
`user-UID.slice`. One drop-in for the user manager does that at every login; `scripts/shura-dmem root-part` prints it:

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
sudo tee /etc/systemd/system/user@.service.d/shura-dmem.conf <<'EOF'
# Shura (MEMV2): let the user manager use the dmem controller (device memory) for its slices.
# Remove this file and run `systemctl daemon-reload` to undo; the controller goes away at the next login.
[Service]
ExecStartPre=+/bin/sh -c 'echo +dmem > /sys/fs/cgroup/user.slice/cgroup.subtree_control; echo +dmem > /sys/fs/cgroup/user.slice/user-%i.slice/cgroup.subtree_control'
EOF
sudo systemctl daemon-reload      # then log out and back in
```

Undo: `sudo rm /etc/systemd/system/user@.service.d/shura-dmem.conf && sudo systemctl daemon-reload`, then log out and in.

**2. Shura's own slice (user, part of a deploy).**

```bash
mkdir -p ~/.config/systemd/user/ai-gateway.service.d
printf '[Service]\nSlice=shura.slice\n' > ~/.config/systemd/user/ai-gateway.service.d/shura-slice.conf
systemctl --user daemon-reload && systemctl --user restart ai-gateway
```

Undo: remove the file, `systemctl --user daemon-reload && systemctl --user restart ai-gateway`.

**3. The protection (user, no root).** `scripts/shura-dmem apply` enables dmem inside the user manager and writes
`dmem.low = max` for `session.slice` and `app.slice`; `scripts/shura-dmem undo` sets it back to 0. The values do not survive a
logout: a user unit that runs `shura-dmem apply` at login can do it (not installed by default).

## Checking it on the stand

1. `cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/{session,app,shura}.slice/dmem.current`: each slice
   accounts its VRAM.
2. Overcommit the card on purpose (the stand's VRAM eater in a scope under `app.slice`, the model under `shura.slice`): Shura's
   `dmem.current` falls, the desktop's does not; the governor's journal and the server's log show the freeze and its end.
3. The VRAM watchdog of the stand stays on for this test.
