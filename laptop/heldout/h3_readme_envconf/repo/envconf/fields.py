_MISSING = object()

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_bool(text):
    """'1/true/yes/on' -> True, '0/false/no/off' -> False (регистр не важен).

    Всё остальное (в том числе пустая строка) -> ValueError.
    """
    t = text.strip().lower()
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    raise ValueError(f"not a bool: {text!r}")


class Field:
    """Описание одной настройки.

    type    -- int, float, str, bool или list (для list значение режется по sep)
    default -- значение по умолчанию; если не задано, переменная обязательна
    secret  -- скрывать значение в Config.as_dict()
    sep     -- разделитель для type=list (по умолчанию ",")
    """

    def __init__(self, type=str, default=_MISSING, *, secret=False, sep=","):
        if type not in (int, float, str, bool, list):
            raise TypeError(f"unsupported type: {type!r}")
        self.type = type
        self.default = default
        self.secret = secret
        self.sep = sep

    @property
    def required(self):
        return self.default is _MISSING

    def convert(self, raw):
        if self.type is bool:
            return parse_bool(raw)
        if self.type is list:
            return [p.strip() for p in raw.split(self.sep) if p.strip()]
        return self.type(raw)
