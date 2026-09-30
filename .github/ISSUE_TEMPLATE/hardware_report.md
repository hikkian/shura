---
name: Hardware report
about: Tell us how Shura behaves on your hardware (this is the most useful thing you can send)
title: "[hardware] <GPU or CPU model> on <OS>"
labels: hardware-report
---

**1. Run this and paste the result below** (it writes `shura-hardware-report.md`, nothing is sent anywhere). If you ran
`shura install`, the same file is already in the Shura folder and includes the speed we predicted and the speed measured:

```bash
shura report
```

<!-- Paste the content of shura-hardware-report.md here. It has no names, paths or IDs, but please read it first. -->

**2. What actually happened** (optional but very valuable)

- Did the install work? Which backend ended up being used (cuda, rocm, vulkan, sycl, metal, cpu)?
- Real speed you saw (tokens/s for generation) and context size:
- Anything that failed or looked wrong (if the install stopped, please attach `logs/verify.log` from the Shura folder:
  `~/.local/share/shura`, `~/Library/Application Support/shura` or `%LOCALAPPDATA%\shura`):
