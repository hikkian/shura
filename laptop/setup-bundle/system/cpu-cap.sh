#!/usr/bin/env bash
# CPU package power limit (Intel RAPL) for cooler LLM runs. Not persistent: a reboot restores the firmware values.
#   sudo ./cpu-cap.sh 45        long-term limit PL1 = 45 W, short boost PL2 = PL1 + 15 W
#   sudo ./cpu-cap.sh 45 45     PL1 = 45 W and PL2 = 45 W (no hot boost bursts)
#   sudo ./cpu-cap.sh reset     back to the factory values (PL1 70 W, PL2 162 W)
#   ./cpu-cap.sh                show the current limits
set -euo pipefail
P=/sys/class/powercap/intel-rapl:0
show() { echo "PL1 $(( $(cat $P/constraint_0_power_limit_uw) / 1000000 )) W, PL2 $(( $(cat $P/constraint_1_power_limit_uw) / 1000000 )) W"; }
case "${1:-}" in
  "") show ;;
  reset) for d in $P /sys/class/powercap/intel-rapl-mmio:0; do [ -w $d/constraint_0_power_limit_uw ] || continue
           echo 70000000 > $d/constraint_0_power_limit_uw; echo 162000000 > $d/constraint_1_power_limit_uw; done; show ;;
  *[!0-9]*) echo "usage: sudo $0 <watts> [pl2-watts]|reset"; exit 1 ;;
  *) w=$1; b=${2:-$(( $1 + 15 ))}; case "$b" in *[!0-9]*|"") echo "usage: sudo $0 <watts> [pl2-watts]"; exit 1 ;; esac
     [ "$w" -ge 25 ] && [ "$w" -le 70 ] || { echo "limit must be 25..70 W"; exit 1; }
     [ "$b" -ge "$w" ] && [ "$b" -le 162 ] || { echo "PL2 must be PL1..162 W"; exit 1; }
     echo $(( w * 1000000 )) > $P/constraint_0_power_limit_uw
     echo $(( b * 1000000 )) > $P/constraint_1_power_limit_uw
     # MMIO copy of the limit (some laptops enforce the lower of MSR and MMIO): keep them equal if present
     M=/sys/class/powercap/intel-rapl-mmio:0
     if [ -w $M/constraint_0_power_limit_uw ]; then
       echo $(( w * 1000000 )) > $M/constraint_0_power_limit_uw
       echo $(( b * 1000000 )) > $M/constraint_1_power_limit_uw
     fi
     show ;;
esac
