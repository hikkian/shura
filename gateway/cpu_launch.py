"""Set the inherited initial courtesy level before exec; used only on reversible hosts."""
import os
import sys


def launch(level, argv):
    if level not in range(4) or not argv:
        raise ValueError("invalid courtesy level or empty server command")
    os.setpriority(os.PRIO_PROCESS, 0, 10 if level == 0 else 19)
    os.sched_setscheduler(0, os.SCHED_IDLE if level >= 2 else os.SCHED_OTHER, os.sched_param(0))
    if level == 3:
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(cpus[:max(1, len(cpus)//2)]))
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    launch(int(sys.argv[1]), sys.argv[2:])
