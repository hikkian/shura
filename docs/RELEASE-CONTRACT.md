# Договор о выпуске shura.cpp: сборка и установщик

Версия договора: schema 1. Образец, карточка RELBIN1. Код, который его исполняет:
`installer/universal/release_manifest.py` (проверка и выбор), `installer/universal/backends.py` (`parse_release_asset`,
`release_asset_name`), `installer/universal/engine.py` (`install_manifest_release`). Проверяют: `tests/test_release_contract.py`,
`tests/test_release_manifest.py`.

## Один файл на выпуск

В каждом выпуске лежит ровно один `release-manifest.json` рядом с архивами. Его адрес: `SHURA_RELEASE_MANIFEST` или
`release_manifest_url` в `catalog/installer.json` (только https).

## Имя архива (одно для манифеста и установщика)

    shura-cpp-<версия>-<семейство>-<os>-<arch>.tar.gz

- версия: цифры, точки, `+ ~` и буквы без дефиса, начинается с цифры (`1.0.0`, `1.0.0rc1`);
- семейство строчными: `hip`, `cuda`, `vulkan`, `sycl`, `cpu`;
- os: `linux`; arch: `x86_64` или `arm64`;
- только `.tar.gz`; имя простое: без `/`, `\`, `..`, точки в начале, NUL.

Пример: `shura-cpp-1.0.0-hip-linux-x86_64.tar.gz`. Разбор один: `backends.parse_release_asset(name)` (возвращает
version, family, backend, os, arch, либо None). Имена апстрима `llama-bNNNN-bin-...` разбирает отдельно `parse_asset`, это другая ветка.
Имя в `assets` обязано совпасть с `family`, `version` и `arch` самой записи, иначе манифест отвергается.

## Манифест

```json
{
  "schema": 1,
  "releases": [{
    "id": "shura-1.0.0-hip-linux-x86_64",
    "engine": "shura.cpp",
    "version": "1.0.0",
    "family": "HIP",
    "arch": "x86_64",
    "gpu_targets": ["gfx1201", "gfx1200"],
    "runtime": {"rocm_min": "6.2", "rocm_max": "7.9"},
    "glibc_min": "2.31",
    "parts": ["vram_governor", "gqa_kernel", "mtp"],
    "built_with": {
      "commit": "<git sha shura.cpp>",
      "cmake_flags": ["-DGGML_HIP=ON", "-DAMDGPU_TARGETS=gfx1200;gfx1201"],
      "compiler": "clang 19.1.0",
      "runtime": {"rocm": "7.2.4"}
    },
    "assets": [{"name": "shura-cpp-1.0.0-hip-linux-x86_64.tar.gz", "url": "https://...", "sha256": "<64 hex>", "size": 123456789}]
  }]
}
```

Обязательные поля записи: `id`, `engine` (`shura.cpp`), `version`, `family`, `gpu_targets`, `runtime`, `glibc_min`, `parts`,
`built_with` (внутри обязателен `commit`), `assets` (в каждом `name`, `url` https, `sha256`; `size` целое, если есть).
Необязательные: `arch` (по умолчанию `x86_64`), `cpu_isa`.

### Семейства и цели видеокарт

| Семейство | gpu_targets | Среда в `runtime` |
|---|---|---|
| HIP | gfx-коды (`gfx1201`), не пусто | `rocm_min`, `rocm_max` |
| CUDA | compute capability (`8.9`, `12.0`), не пусто | `nvidia_driver_min`, `nvidia_driver_max` |
| Vulkan | всегда `[]` | `vulkan_min` |
| SYCL | всегда `[]` | `oneapi_min` |
| CPU | всегда `[]` | пусто |

Для HIP и CUDA хотя бы один минимум или максимум обязателен (выбор не гадает). Версии сравниваются как числа через точку.

### Наши части (`parts`)

Список имён наших частей, которые работают в этой сборке: `vram_governor`, `gqa_kernel`, `mtp` и другие. Установщик по нему
пишет в отчёте, что включено. Выбор сборки от него не зависит.

### Сборка CPU и наборы команд

`cpu_isa` (например `["avx2", "avx512f", "avx512_vnni"]`) только сведения для человека и отчёта. Выбор сборки не меняется.

## Правила выбора на стороне установщика

Сборка берётся, только если подходят все условия: движок `shura.cpp`; ОС Linux; архитектура; семейство есть среди backends
машины; для HIP и CUDA цель машины входит в `gpu_targets`; версия среды в [min, max]; glibc машины не меньше `glibc_min`.
Любое неизвестное значение машины считается «не подходит». Если не подходит ни одна, установщик пишет причину по каждой
записи (какое условие не совпало) и идёт в сборку из исходников, затем в апстрим. Среди подходящих: порядок backends машины, затем
новейшая версия.

Сумма не сошлась: SHA-256 проверяется до распаковки, при расхождении файл удаляется, распаковки нет, эта сборка отвергнута.
Другая сборка манифеста не пробуется, сразу сборка из исходников. Имя архива не по договору (в том числе `../x`): манифест отвергается
целиком до любой загрузки.

## Что должна делать сборка (сторона выпуска)

Положить в выпуск архивы по имени выше, посчитать `sha256` и `size`, заполнить `parts` и `built_with` по факту сборки
(не по желанию), записать один `release-manifest.json`. Архив содержит `llama-server` (поиск по дереву, как в `engine.find_server`).
