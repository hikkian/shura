---
name: Hardware report
about: Tell us how Shura behaves on your hardware (this is the most useful thing you can send)
title: "[hardware] <GPU or CPU model> on <OS>"
labels: hardware-report
---

**1. Run this and paste the result below** (it writes `shura-hardware-report.md`, nothing is sent anywhere):

```bash
shura report
```

<!-- Paste the content of shura-hardware-report.md here. It has no names, paths or IDs, but please read it first. -->

**2. What actually happened** (optional but very valuable)

- Did the install work? Which backend ended up being used (cuda, rocm, vulkan, sycl, metal, cpu)?
- Real speed you saw (tokens/s for generation) and context size:
- Anything that failed or looked wrong:
