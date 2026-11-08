import argparse
import sys

TASKS = [
    (1, "Купить молоко", False),
    (2, "Позвонить маме", True),
    (3, "Write report", False),
]


def cmd_list(args):
    for tid, title, done in TASKS:
        mark = "x" if done else " "
        print(f"[{mark}] {tid}. {title}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="todo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
