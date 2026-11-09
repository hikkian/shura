import os

from .errors import ConfigError
from .fields import Field

MASK = "***"


class Config:
    """Результат load(). Доступ только через атрибуты: cfg.port (cfg["port"] не работает)."""

    def __init__(self, values, secrets):
        self.__dict__["_values"] = dict(values)
        self.__dict__["_secrets"] = frozenset(secrets)

    def __getattr__(self, name):
        try:
            return self.__dict__["_values"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AttributeError("Config is read-only")

    def as_dict(self, reveal=False):
        """Словарь настроек. Секретные значения заменены на '***', если не reveal=True."""
        return {k: (MASK if (k in self._secrets and not reveal) else v) for k, v in self._values.items()}


def load(schema, *, prefix, environ=None):
    """Читает настройки по schema = {имя: Field | тип}.

    Имя переменной окружения: PREFIX_ИМЯ в верхнем регистре, например prefix="app", имя "port" -> APP_PORT.
    prefix обязателен и передаётся только по имени. environ по умолчанию os.environ.
    Пустая строка в переменной считается отсутствующей (берётся default или ошибка).
    Если есть проблемы, бросает ConfigError со всеми ошибками сразу.
    """
    env = os.environ if environ is None else environ
    values, secrets, errors = {}, [], []
    for name, spec in schema.items():
        field = spec if isinstance(spec, Field) else Field(spec)
        var = f"{prefix}_{name}".upper()
        raw = env.get(var, "")
        if raw == "":
            if field.required:
                errors.append(f"{var}: missing")
            else:
                values[name] = field.default
        else:
            try:
                values[name] = field.convert(raw)
            except ValueError as e:
                errors.append(f"{var}: {e}")
        if field.secret:
            secrets.append(name)
    if errors:
        raise ConfigError(errors)
    return Config(values, secrets)
