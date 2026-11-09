# envconf

Маленькая библиотека для чтения настроек приложения из переменных окружения. Вы описываете схему (имя настройки, тип, значение по умолчанию), а библиотека читает переменные, приводит значения к нужным типам и, если что-то не так, сообщает сразу обо всех проблемах, а не только о первой.

Зависимостей нет, нужен Python 3.9 или новее. Библиотека лежит в каталоге `envconf/`, подключается обычным импортом:

```python
from envconf import load, Field, Config, ConfigError, parse_bool
```

## Как формируется имя переменной

Имя переменной окружения строится как `PREFIX_ИМЯ` в верхнем регистре. Для `prefix="app"` и настройки `port` будет прочитана переменная `APP_PORT`. Пустая строка в переменной считается отсутствующим значением: берётся значение по умолчанию либо возникает ошибка.

## load

`load(schema, *, prefix, environ=None)` читает настройки по схеме. Схема — словарь, где значение это `Field` или просто тип (`int`, `str` и т. д.). Аргумент `prefix` обязателен и передаётся только по имени. Аргумент `environ` необязателен: по умолчанию используется `os.environ`, но в тестах удобно передать обычный словарь. Функция возвращает объект `Config`.

```python
from envconf import load, Field

schema = {
    "port": int,
    "debug": Field(bool, False),
}
cfg = load(schema, prefix="app", environ={"APP_PORT": "8080"})
print(cfg.port, cfg.debug)  # 8080 False
```

## Field

`Field(type=str, default=..., *, secret=False, sep=",")` описывает одну настройку. Поддерживаются типы `int`, `float`, `str`, `bool` и `list`; любой другой тип вызовет `TypeError`. Если `default` не указан, настройка обязательна. Параметр `secret=True` скрывает значение в `Config.as_dict()`. Для `list` значение режется по разделителю `sep`, пробелы по краям убираются, пустые элементы отбрасываются.

```python
from envconf import load, Field

schema = {
    "hosts": Field(list, sep=";"),
    "ratio": Field(float, 0.5),
}
cfg = load(schema, prefix="svc", environ={"SVC_HOSTS": "a.example; b.example;"})
print(cfg.hosts)  # ['a.example', 'b.example']
print(cfg.ratio)  # 0.5
```

## Config

`Config` — результат `load()`. Он только для чтения: значения доступны как атрибуты (`cfg.port`), а запись или обращение вида `cfg["port"]` не работают. Метод `as_dict(reveal=False)` возвращает словарь настроек; значения секретных полей заменяются на `***`, если не передать `reveal=True`.

```python
from envconf import load, Field

cfg = load({"token": Field(str, secret=True), "name": Field(str, "demo")},
           prefix="myapp", environ={"MYAPP_TOKEN": "s3cret"})
print(cfg.as_dict())             # {'token': '***', 'name': 'demo'}
print(cfg.as_dict(reveal=True))  # {'token': 's3cret', 'name': 'demo'}
```

## ConfigError

Если обязательная переменная не задана или значение не приводится к типу, `load()` бросает `ConfigError`. В атрибуте `errors` лежит список всех найденных проблем сразу, например `"APP_PORT: missing"`.

```python
from envconf import load, ConfigError

try:
    load({"port": int, "workers": int}, prefix="app", environ={"APP_WORKERS": "много"})
except ConfigError as e:
    for line in e.errors:
        print(line)
```

## parse_bool

`parse_bool(text)` превращает строку в булево значение: `1`, `true`, `yes`, `on` дают `True`; `0`, `false`, `no`, `off` дают `False`, регистр не важен. Всё остальное, включая пустую строку, вызывает `ValueError`.

```python
from envconf import parse_bool

print(parse_bool("Yes"))   # True
print(parse_bool("off"))   # False
try:
    parse_bool("maybe")
except ValueError as e:
    print("ошибка:", e)
```
