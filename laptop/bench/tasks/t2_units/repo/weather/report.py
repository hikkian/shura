from weather.units import f_to_c


def format_reading(city, temp_f):
    return f"{city}: {round(f_to_c(temp_f), 1)}°C"


def report(readings):
    return "\n".join(format_reading(c, t) for c, t in readings)
