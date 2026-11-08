"""Подсчёт оценок учеников."""


def average(grades):
    """Средний балл по списку оценок."""
    return sum(grades) / len(grades)


def letter(avg):
    if avg >= 4.5:
        return "A"
    if avg >= 3.5:
        return "B"
    if avg >= 2.5:
        return "C"
    return "D"


def build_rows(students):
    """Список (имя, средний балл, буква), отсортированный по среднему баллу по убыванию."""
    rows = []
    for s in students:
        avg = average(s["grades"])
        rows.append((s["name"], avg, letter(avg)))
    rows.sort(key=lambda r: -r[1])
    return rows
