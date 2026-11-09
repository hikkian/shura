class ConfigError(Exception):
    """Ошибка конфигурации. Собирает ВСЕ проблемы сразу, а не только первую.

    .errors -- список строк вида "APP_PORT: not an int: 'abc'".
    """

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))
