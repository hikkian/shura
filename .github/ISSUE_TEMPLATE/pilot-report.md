---
name: Pilot report
about: Join the pilot programme: tell us what Shura did on your machine, so it goes into docs/COMPATIBILITY.md
title: "[pilot] <GPU> on <OS>"
labels: pilot
---

Thank you for testing. The table in `docs/COMPATIBILITY.md` is built from your report file, so please attach the file itself.

**1. Create the report** (nothing is sent anywhere; it writes `shura-hardware-report.md` into the Shura folder):

```bash
./scripts/shura report
```

Open the file and check it: it should have no names, paths or IDs. Attach it to this issue.

**2. What you ran**

- The command (`./scripts/shura install`, or `./scripts/shura check` if you only wanted the plan):
- Did the server start and answer? (yes / no)

**3. Speed you saw** (tokens per second for generation, and the context size you used):

**4. Anything that looked wrong** (a warning, a plan that was too slow, a failed step). Quote the line.

**5. Your driver and OS**, if the report does not show them (driver version, distribution and version):
