import json

from grades import build_rows


def main():
    with open("students.json", encoding="utf-8") as f:
        students = json.load(f)
    for name, avg, letter in build_rows(students):
        print(f"{name:<12} {avg:5.2f}  {letter}")


if __name__ == "__main__":
    main()
