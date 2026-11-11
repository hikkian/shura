# Инструкция: поднять Шуру на ноутбуке Жомарта (копия нашей настройки)

Для ИИ-агента, который будет выполнять инструкцию и не видел нашего контекста. Читай целиком до начала. Выполняй шаги по порядку, после каждого шага запускай проверку и не иди дальше, пока она не прошла.

Обозначения у каждой команды и проверки:
- **[ПРОВЕРЕНО]** запускали у себя, результат указан;
- **[НЕ ПРОВЕРЕНО]** написано по документации или по аналогии, запускать с вниманием, сверять вывод;
- `$HOME` везде означает домашний каталог Жомарта; у нас это `/home/hikkian`, и в файлах из `setup-bundle/` этот путь автоматически заменяется скриптом `apply-bundle.sh`.

## 0. Что такое Шура и что мы собираем

Шура это локальный ИИ-программист (агент в терминале, без облака): агент **ShuraCode** → **шлюз** `ai_gateway.py` (порт 8080, сам загружает и выгружает модель) → **llama-server** (порт 8090, наш форк llama.cpp `shura.cpp`) → модель **Ornith-1.5 35B-A3B** (3-битная, с MTP и точечно улучшенным слоем 39). Плюс локальный поиск SearXNG (порт 8888). Всё на одном ноутбуке.

**Железо Жомарта (как у нас):** 24 ГБ ОЗУ, NVIDIA RTX 3050 6 ГБ (CUDA-архитектура `sm_86`), Fedora, GNOME. У нас Fedora 44 с KDE; версия Fedora и рабочий стол влияют только на пункты, помеченные «GNOME».

**Где лежат наши файлы:** репозиторий `https://github.com/hikkian/shura`, ветка `laptop`. В нём каталог `laptop/setup-bundle/` содержит **все ручные правки**, которых нет в обычных репозиториях (без них Шура не соберётся «как у нас»): шлюз, конфиги, шаблоны, подсказка v6, плагин, патч движка. Скрипт `laptop/setup-bundle/apply-bundle.sh` раскладывает их по местам.

### 0.1. Сколько места и времени (замеряно у нас, кроме отмеченного)

| Что | Размер |
|---|---|
| Исходные файлы моделей для скачивания: база 3bit 12,99 ГБ, Q8_0 37,80 ГБ, шард с MTP-головой 4,38 ГБ, проектор зрения 0,61 ГБ | 55,8 ГБ |
| Промежуточные: донор Q4_K 20,06 ГБ, голова MTP 2,1 ГБ, база с MTP 13,38 ГБ | 35,5 ГБ |
| Итоговый файл модели `l39p` + проектор | **14,2 ГБ** (остаётся после чистки) |
| Контейнер сборки (образ CUDA 13.0.2 devel) + исходники движка + результат сборки | ~11 ГБ [НЕ ПРОВЕРЕНО итоговый размер каталога сборки] |
| **Пик диска, если удалять по ходу, как написано в шаге 3** | **около 75 ГБ для моделей** + 11 ГБ движок = **≥ 90 ГБ свободно** |
| Если НЕ удалять промежуточное | около 105 ГБ только моделей |

Время:
- **Скачивание** на нашем канале 3,3 МБ/с: Q8_0 3,2 ч, база 1,1 ч, шард 0,4 ч, всё около 4,7 ч. У Жомарта скорость другая, считай по своей. Скачивание возобновляется (`curl -C -`).
- Сборка модели после скачивания: голова MTP 14 с, квантизация головы 2 с, слияние 29 с, донор Q4_K около 6 мин, подмена слоя 39 около 3 мин, проверка sha256 около 1–2 мин на файл. Итого около 15 минут.
- Сборка движка в контейнере: [НЕ ПРОВЕРЕНО полностью, ожидай 40–90 минут и 10+ ГБ ОЗУ при `-j4`; конфигурация и первая цель проверены, см. шаг 2].
- Установка драйвера и CUDA, ShuraCode, поиска: ~1 час без учёта скачиваний.

### 0.2. Общие правила
- Не используй `sudo` там, где не написано. Права нужны для `dnf`, драйвера, `/etc`.
- Не запускай два тяжёлых процесса (сборка движка, квантизация, модель) одновременно: ОЗУ 24 ГБ.
- Секретов в инструкции и в `setup-bundle/` нет; если найдёшь что-то похожее на ключ, не публикуй.
- Все пути, указанные как `~/...`, раскрываются в `$HOME`.

---

## 1. Драйвер NVIDIA и CUDA на Fedora

**Что стоит у нас (факты):** Fedora 44, ядро `7.2.15-200.fc44`, драйвер `akmod-nvidia-615.78.08` из RPM Fusion nonfree, `cuda-toolkit-13-1-13.1.2` из репозитория NVIDIA (у нас подключён `cuda-fedora42`), каталог `/usr/local/cuda -> cuda-13.1`, `nvcc 13.1.115`. Режим графики гибридный (Intel + NVIDIA).

### 1.1. Репозитории и драйвер [НЕ ПРОВЕРЕНО на чистой системе; версии пакетов у нас подтверждены]
```bash
sudo dnf install -y https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
                    https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
# подождать 5-10 минут, пока akmods соберёт модуль ядра; проверить:
modinfo -F version nvidia   # ожидается 615.78.08 или новее
```
Если включён Secure Boot: до перезагрузки выполнить `sudo kmodgenca -a && sudo mokutil --import /etc/pki/akmods/certs/public_key.der`, перезагрузиться и подтвердить ключ в синем экране MOK. Без Secure Boot этого не нужно.
Перезагрузись: `sudo reboot`.

### 1.2. CUDA Toolkit 13.1 [НЕ ПРОВЕРЕНО на чистой системе; репозиторий отвечает 200: проверено]
```bash
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/cuda-fedora42.repo
sudo dnf install -y cuda-toolkit-13-1
ls -d /usr/local/cuda*          # ожидается: /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.1
```
Если команда `dnf config-manager addrepo` не найдена, поставь `sudo dnf install -y dnf-plugins-core` и используй `sudo dnf config-manager --add-repo <URL>`.
Репозиторий `fedora44` с того же сайта тоже отвечает (проверено `curl -I`: 200), но мы ставили из `fedora42`.
Хост-компилятор: на Fedora 44 gcc 16, **nvcc 13.1 с ним не компилирует** (проверено: `char8_t is undefined` в `type_traits`), поэтому движок собирается в контейнере (шаг 2). На хосте компилятор CUDA не нужен, нужны только библиотеки выполнения (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`), они приходят с `cuda-toolkit-13-1`.

### 1.3. Проверка шага 1
```bash
nvidia-smi
```
Ожидается: таблица с `NVIDIA GeForce RTX 3050 ...`, `Memory-Usage: ... / 6144MiB`, `Driver Version: 615.78.08` (или новее), `CUDA Version: 13.x`. [ПРОВЕРЕНО у нас: 615.78.08 / CUDA UMD 13.4 / 6144 MiB.]
```bash
/usr/local/cuda/bin/nvcc --version | tail -2        # release 13.1, V13.1.115
ls /usr/local/cuda/lib64 | grep -E 'libcudart.so.13|libcublas.so.13|libcublasLt.so.13'   # три файла
```
**Откат шага 1:** `sudo dnf remove 'cuda-*' 'akmod-nvidia' 'xorg-x11-drv-nvidia*' 'kmod-nvidia*' && sudo reboot`. Драйвер можно оставить.

### 1.4. Прочие пакеты (нужны дальше)
```bash
sudo dnf install -y git cmake gcc gcc-c++ podman python3 python3-pip nodejs npm curl ImageMagick tesseract tesseract-langpack-rus hunspell
python3 --version; node --version; npm --version; podman --version     # у нас: 3.14.10, нужен node/npm любой свежий LTS, podman 5.8.9
tesseract --list-langs | grep -E '^(eng|rus)$'                        # обе строки
```
[НЕ ПРОВЕРЕНО установка одной командой; пакеты и версии у нас подтверждены: ImageMagick, tesseract 5.5.4 с rus, hunspell, podman 5.8.9.]
Русский словарь для проверки опечаток (hunspell в Fedora его не ставит как отдельный пакет у нас, мы положили файлы вручную) [ПРОВЕРЕНО: скачано и работает]:
```bash
mkdir -p ~/.local/share/hunspell && cd ~/.local/share/hunspell
curl -fsSL -o ru_RU.dic https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/ru/index.dic
curl -fsSL -o ru_RU.aff https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/ru/index.aff
echo "Алаты монторинге прверить коммит" | tr ' ' '\n' | hunspell -d ru_RU -l      # выводит все 4 слова как неизвестные
```
Автор словаря: wooorm/dictionaries (лицензия смотри в репозитории), в наш репозиторий мы файлы НЕ кладём.

---

## 2. Движок llama.cpp (форк shura.cpp) с CUDA

**Что собираем:** репозиторий `https://github.com/hikkian/shura.cpp`, ветка `shura`, коммит **`c4d92d5e4d22d6aff9e4d7779b9e8a0bf75763b7`** (30.10.2026) **плюс наш патч из 7 файлов без коммита** (`setup-bundle/engine/shura.cpp-uncommitted.patch`: 229 добавлений, 25 удалений; затрагивает `common/arg.cpp`, `common/common.cpp`, `common/common.h`, `common/sampling.cpp`, `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, `ggml/src/ggml-cuda/fattn-vec.cuh`, `tools/server/server-schema.cpp`). [ПРОВЕРЕНО: патч применяется к чистому checkout этого коммита без ошибок.]
**Внимание:** `setup.sh` из репозитория `hikkian/shura` собирает ДРУГОЙ движок (`thecodacus/llama.cpp`, коммит `27c54b4bbcefadedcec6397477cc2e866c1db716`). Его запускать НЕ нужно. Из него нам нужен только `llama-quantize` (шаг 3.5), см. там.

### 2.1. Исходники
```bash
mkdir -p ~/.local/share/shura && cd ~/.local/share/shura
git clone --branch shura https://github.com/hikkian/shura.cpp.git shura.cpp
cd shura.cpp && git checkout c4d92d5e4d22d6aff9e4d7779b9e8a0bf75763b7
git rev-parse HEAD                                    # c4d92d5e4d22d6aff9e4d7779b9e8a0bf75763b7
# патч берём из ветки laptop репозитория hikkian/shura (шаг 5.1 клонирует её в ~/shura; если ещё нет, сделай шаг 5.1 сейчас)
git apply --check ~/shura/laptop/setup-bundle/engine/shura.cpp-uncommitted.patch && git apply ~/shura/laptop/setup-bundle/engine/shura.cpp-uncommitted.patch
git diff --stat | tail -1                             # 7 files changed, 229 insertions(+), 25 deletions(-)
```

### 2.2. Сборка в контейнере (как у нас: Ubuntu 24.04, gcc 13.3.0, cmake 3.28.3, CUDA 13.0.88)
Образ: `docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04` (7,4 ГБ).
```bash
cd ~/.local/share/shura/shura.cpp
podman pull docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04
podman run --rm -v "$PWD:/src:Z" docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 bash -c '
  set -e; export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq cmake build-essential libssl-dev
  cmake -S /src -B /src/build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
  cmake --build /src/build-cuda -j4 --target llama-server llama-quantize llama-perplexity llama-bench'
ls -la build-cuda/bin/llama-server
```
Флаги, которые стоят у нашей рабочей сборки (из `CMakeCache.txt`): `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, `GGML_NATIVE=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_GRAPHS=ON`, `GGML_OPENMP=ON`, `GGML_CPU_REPACK=ON`, `GGML_AVX=OFF` (при `GGML_NATIVE=ON` выбор инструкций берётся из процессора), `LLAMA_CURL=OFF`, `LLAMA_OPENSSL=ON`, `GGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;f16-turbo2_0;f16-turbo3_0;f16-turbo4_0;q8_0-turbo2_0;q8_0-turbo3_0;q8_0-turbo4_0` (это значение по умолчанию репозитория, отдельно задавать не нужно).
[ПРОВЕРЕНО: в этом же образе команда `cmake -S /src -B /src/build-cuda ...` настраивает без ошибок (cmake 3.28.3, gcc 13.3.0, nvcc 13.0.88, OpenSSL 3.0.13), цель `ggml-base` собирается. Полная сборка `llama-server` с ядрами CUDA НЕ запускалась (долго); если упрётся в нехватку ОЗУ, замени `-j4` на `-j2`.]
Результат: `~/.local/share/shura/shura.cpp/build-cuda/bin/llama-server` и `llama-quantize` (у нас `llama-server --version`: `0.7.0-dev (build 11918, commit c4d92d5e4)`; **наш рабочий `llama-quantize` для донора был из другой сборки** `llama.cpp-perf`, см. 3.5).

### 2.3. Проверка сборки
```bash
B=~/.local/share/shura/shura.cpp/build-cuda/bin
LD_LIBRARY_PATH=$B:/usr/local/cuda/lib64 $B/llama-server --version      # version: ... commit c4d92d5e4
LD_LIBRARY_PATH=$B:/usr/local/cuda/lib64 ldd $B/llama-server | grep -E "not found" || echo "все библиотеки найдены"
```
**Откат шага 2:** `rm -rf ~/.local/share/shura/shura.cpp/build-cuda && git -C ~/.local/share/shura/shura.cpp checkout -- . && git -C ~/.local/share/shura/shura.cpp clean -fd` (вернуть исходники к чистому коммиту), контейнер: `podman rmi docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04`.

---

## 3. Модель: собрать самому

**Что получится:** файл `Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP-l39p.gguf` (13 570 606 784 байт), sha256 `f042eb2c2bc832ae74a01148caff317dcc8b0ada23dd808349cd012c9649afda`. Это 3-битная модель Ornith (автор квантизации pfeifferj) с MTP-головой (ускорение генерации на 40%) и экспертами последнего слоя 39 в 4 битах (меньше опечаток).
[ПРОВЕРЕНО: весь рецепт ниже, шаги 3.3-3.6, мы заново выполнили на нашей машине, и итоговые файлы совпали с рабочими **бит в бит** по sha256: голова, база с MTP и итог.]

### 3.1. Скачать файлы (Hugging Face, токен не нужен)
Все ссылки зафиксированы на конкретные ревизии репозиториев. Кладём в `~/.local/share/shura/models/` (создай). Скачивание возобновляемое.
```bash
M=~/.local/share/shura/models; mkdir -p $M; cd $M
# 1) база 3bit (12 992 461 216 байт), sha256 9cae3b59a34308664b53e5b80fd74914cd0c1cf504bcd05609498b0f9126a74f
curl -fL -C - -o Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf \
  https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF/resolve/0bfdfe7f6c424de60481c26379c9d821469f75f1/Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf
# 2) шард с MTP-головой (4 378 994 104 байт), sha256 63d592b6c5efb743e521fc406282353671f5f8d60540dd6017074613b5143f9f
curl -fL -C - -o model-00016-of-00016.safetensors \
  https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/resolve/10fbf86fed7ecee4a061f8b499a618f46001cac1/model-00016-of-00016.safetensors
# 3) проектор зрения Q8_0 (614 194 176 байт), sha256 2d1f84623497449e5fe085836072d1e5db9ccac3e36ebbbf4ac573016cc65053
curl -fL -C - -o mmproj-Q8_0.gguf \
  https://huggingface.co/IsValorum/Tiel-Coder-35B-A3B-APEX-I-NanoPlus-GGUF/resolve/89a0b41aa70f275edf19ad203a9ad5714947d704/mmproj-Q8_0.gguf
sha256sum Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf model-00016-of-00016.safetensors mmproj-Q8_0.gguf
```
[ПРОВЕРЕНО: имена, размеры и sha256 взяты из API Hugging Face (`lfs.sha256`) и совпадают с нашими локальными файлами; ссылки отвечают.] Если sha256 не совпал, перекачай файл (`rm` и `curl` заново).
Проектор Q8_0 в репозитории Tiel, но это тот же зрительный блок, что у Ornith (проверено: последние 800 МБ BF16-версий совпадают); у Ornith есть своя BF16-версия (0,9 ГБ), мы её НЕ используем: она тратит на ~1 ГБ больше ОЗУ.
**Q8_0 (37,8 ГБ) пока НЕ качай:** он нужен только для донора на шаге 3.5, а качать его лучше после слияния (пик диска).

### 3.2. Подготовка Python и инструментов [ПРОВЕРЕНО]
Нужен Python с `numpy` и рабочий `gguf-py`. **`gguf-py` из `shura.cpp` у нас не импортируется** (SyntaxError в `gguf/tensor_mapping.py:3084`), поэтому в бандле лежит рабочая копия, которой мы пользовались (gguf 0.19.0, `~/shura/laptop/setup-bundle/gguf-py`).
```bash
python3 -m venv ~/shura-venv && ~/shura-venv/bin/pip install numpy
export GGUF_PY=$HOME/shura/laptop/setup-bundle/gguf-py
~/shura-venv/bin/python -c "import sys; sys.path.insert(0,'$GGUF_PY'); import gguf, numpy; print('gguf ok, numpy', numpy.__version__)"
```
(У нас numpy 2.5.4 на Python 3.13; на Python 3.14 ставь `pip install numpy` без версии.) [ПРОВЕРЕНО: скрипт подмены с этой копией gguf-py проходит `--dry-run` на наших файлах.]
Скрипты для следующих шагов лежат в репозитории (шаг 5.1 клонирует его в `~/shura`): `~/shura/scripts/graft_mtp_head.py` (в обычной части репозитория), `~/shura/laptop/tools/weights-fix/splice_gguf.py`, а `graft_unfused.py` лежит у нас в `bench/` и скопирован в бандл: `~/shura/laptop/tools/weights-fix/graft_unfused.py` (после шага 5.1; если файла там нет, проси автора).

### 3.3. Голова MTP [ПРОВЕРЕНО, совпало бит в бит]
```bash
export GGUF_PY=$HOME/shura/laptop/setup-bundle/gguf-py; PY=~/shura-venv/bin/python; M=~/.local/share/shura/models; W=~/shura-work; mkdir -p $W
$PY ~/shura/laptop/tools/weights-fix/graft_unfused.py head $M/Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf $M/model-00016-of-00016.safetensors $W/head-bf16.gguf
sha256sum $W/head-bf16.gguf     # 8cf97f5e75fcb93c372e... (размер 1 701 299 936 байт)
```
Ожидаемый вывод: `20 head tensors for block 40 -> head-bf16.gguf`.

### 3.4. Квантизация головы и слияние с базой [ПРОВЕРЕНО, совпало бит в бит]
Нужен `llama-quantize` (шаг 2.2 собрал его в `shura.cpp/build-cuda/bin/`).
```bash
B=~/.local/share/shura/shura.cpp/build-cuda/bin; export LD_LIBRARY_PATH=$B:/usr/local/cuda/lib64
$B/llama-quantize --allow-requantize --tensor-type 'ffn_.*_exps=q3_k' $W/head-bf16.gguf $W/head-q.gguf q8_0
sha256sum $W/head-q.gguf        # 5ef39f1191cd7140e223... (400 344 800 байт)
$PY ~/shura/scripts/graft_mtp_head.py merge $M/Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf $W/head-q.gguf $M/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf
sha256sum $M/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf
# ожидается ровно: 5d5a06a4e0947beadec8ec6f0e7c9fa793761a3847e9cd16fa946a165db5d381   (13 381 863 104 байт)
```
Ожидаемый вывод слияния: `733 body + 20 head tensors -> ...`. Если sha256 не совпал, проверь версию `llama-quantize` (у нас рабочий был `0.3.0-dev, commit 27c54b4bb` из сборки llama.cpp-perf; результат квантизации головы детерминирован, но другая версия может дать иной файл; тогда сравни хотя бы размер 400 344 800 и продолжай, модель останется рабочей).
Теперь можно удалить: `rm $M/Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf $M/model-00016-of-00016.safetensors $W/head-*.gguf` (освобождает ~19 ГБ).

### 3.5. Донор Q4_K из Q8_0 [ПРОВЕРЕНО у нас; команда та же]
Теперь скачай Q8_0:
```bash
cd $M && curl -fL -C - -o Ornith-1.5-35B-Q8_0.gguf \
  https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q8_0.gguf
sha256sum Ornith-1.5-35B-Q8_0.gguf      # de46c4baf4b4dd85ea438bb0f757f21c38841a353506579979bba114311658c3 (37 802 149 280 байт)
```
Квантизация (чистый Q4_K, `--pure`, это важно: в варианте Q4_K_M часть тензоров получается Q6_K и стоит ~1 ГБ лишней ОЗУ):
```bash
$B/llama-quantize --allow-requantize --pure $M/Ornith-1.5-35B-Q8_0.gguf $W/donor-Q4_K-pure.gguf Q4_K 8
ls -la $W/donor-Q4_K-pure.gguf          # 20 060 898 720 байт, около 6 минут
```
Примечание о версии: у нас донор сделан `llama-quantize` из сборки `llama.cpp-perf` (commit `27c54b4bb`), у тебя будет из `shura.cpp`. Алгоритм Q4_K без imatrix один и тот же, но побайтовое совпадение не гарантируется. Поэтому на шаге 3.7 есть проверка по трём тензорам слоя 39.

### 3.6. Подмена слоя 39 (эксперты gate, up, down) [ПРОВЕРЕНО, совпало бит в бит]
```bash
$PY ~/shura/laptop/tools/weights-fix/splice_gguf.py --base $M/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf \
    --donor $W/donor-Q4_K-pure.gguf --out $M/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP-l39p.gguf \
    --layers 39 --kinds ffn_gate_exps,ffn_up_exps,ffn_down_exps
```
Ожидаемый вывод: таблица с тремя строками `39 ffn_*_exps Q2_K -> Q4_K` (+188 743 680 байт всего, `+0.176 ГиБ`), `САМОПРОВЕРКА: ПРОЙДЕНА`, файл около 12,6 ГиБ (13 570 606 784 байт). У `gguf-py` должен быть путь через `GGUF_PY` (скрипт сам подхватит `~/.unsloth/...` или `GGUF_PY`).

### 3.7. Проверка итога
```bash
sha256sum $M/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP-l39p.gguf
# ожидается: f042eb2c2bc832ae74a01148caff317dcc8b0ada23dd808349cd012c9649afda
```
Если совпало, всё в порядке. **Если не совпало, но размер 13 570 606 784 и скрипт сказал «САМОПРОВЕРКА: ПРОЙДЕНА»**, сравни три тензора слоя 39 (разница может быть только из-за другой версии `llama-quantize` на шаге 3.5):
```bash
$PY - <<'E'
import sys,os,hashlib; sys.path.insert(0,os.environ['GGUF_PY']); import gguf
r=gguf.GGUFReader(os.path.expanduser('~/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP-l39p.gguf'),'r')
for t in r.tensors:
    if t.name.startswith('blk.39.ffn_') and t.name.endswith('_exps.weight'): print(t.name,t.tensor_type.name,int(t.n_bytes),hashlib.sha256(bytes(t.data)).hexdigest())
E
# наши значения (Q4_K, по 150 994 944 байт каждый):
# blk.39.ffn_down_exps.weight  ffb905f8123e0d73cfe17b3a82b4450b73777211d30846fba1bb0149b997da1f
# blk.39.ffn_gate_exps.weight  a5444ae931a7a7921e16acf4a1571bc58f1c332d7de552a9a4e15bc249942b6f
# blk.39.ffn_up_exps.weight    324975cac8314148a795d764f9726a544ffc82b83c154fa196b0877b14f56e6e
```
Несовпадение допустимо (модель всё равно работает), но запиши, что именно не совпало, и сообщи автору.

### 3.8. Чистка и итог
```bash
rm -f $M/Ornith-1.5-35B-Q8_0.gguf $W/donor-Q4_K-pure.gguf
# запасной вариант (откат модели): оставь $M/...-3bit-MTP.gguf (13,4 ГБ); если места мало, удали и его
ls -la $M            # останется: ...-3bit-MTP-l39p.gguf, ...-3bit-MTP.gguf (запас), mmproj-Q8_0.gguf
```
**Если у Жомарта нет места или времени:** можно взять БЕЗ MTP и без слоя 39 только базу 3bit (шаги 3.1 пункт 1, 3.1 пункт 3), поставив в `model-launch.json` `"specType": "none"` (скорость падает с 44 до 32 ток/с) [НЕ ПРОВЕРЕНО на его машине; на нашей скорость без MTP измерена: 32 ток/с]. Это резервный путь, не рекомендуемый.
**Откат шага 3:** удалить созданные файлы; исходные скачанные файлы можно докачать.

---

## 4. Конфиги запуска

Все конфиги лежат в `laptop/setup-bundle/config/` и раскладываются скриптом `apply-bundle.sh` (шаг 5). Ниже что они задают, чтобы агент понимал, что не менять.

**`~/shura/config/model-launch.json`** (параметры запуска llama-server, у модели по умолчанию `ornithi`) [ПРОВЕРЕНО работающий]:
- `ctxSize: 200000` (окно контекста), `parallelSlots: 1`, `nGpuLayers: 99`, `nCpuMoe: 40` (все эксперты на процессоре, остальное на GPU), `flashAttn: on`, `threads: 8`, `threadsBatch: 8`, `batchSize: 2048`, `ubatchSize: 1024`;
- KV-кэш: `cacheTypeK: turbo4`, `cacheTypeV: turbo4` (сжатый кэш, нужен именно наш форк);
- MTP: `specType: draft-mtp`, `specDraftNMax: 2`, `specDraftPMin: 0.5`;
- кэш экспертов: `moeCacheSlots: 6` и в `extraArgs` `--moe-cache-policy two-channel`, профиль `moeCacheProfile` `config/moe-trace/tiel-coder-agentic.csv`;
- `loadMode: none`, `niceLevel: 10` (процесс с пониженным приоритетом, чтобы рабочий стол не тормозил), `alwaysVision: true` (проектор зрения держится в памяти, иначе первое сообщение с картинкой ждёт перезагрузку), `reasoning: auto`, `temperature: 0.4`, `topK: 20`, `topP: 0.95`;
- `extraArgs` содержит `--chat-template-file ~/shura/config/templates/ornith-ideal5.jinja`, `--ban-scripts cjk,homoglyph`, `--ctx-checkpoints`;
- `effortBudgets` (бюджет мыслей по уровням): low 2048, medium 8192, high 16384, xhigh 32768, max -1;
- `langGuard: true` и `typoGuard: true` (защита ответа в шлюзе), `chatTemplateKwargs: {"terse": false}` (профиль `ornithit` с `terse: true` отвечает короче).
- Порты, пороги памяти и простоя: `~/shura/config/guardian.json` (`gatewayPort 8080`, `llamaPort 8090`, `idleUnloadSeconds 900`, `ramFreeMinGB 1.5`, **`ramFreeMinGBToLoad 14.0`**: модель не загрузится, если свободно меньше 14 ГБ ОЗУ; закрой браузер или подожди, `vramGuard false`, `gameMode false`).
- В `model-launch.json` описаны и другие профили (`qwen36*`, `tiel-coder`), чьих файлов у Жомарта нет; они ему не нужны, их не трогай и не запускай.

**`~/.config/shuracode/shuracode.json`** (конфиг ShuraCode): провайдер `tiel-local` (адрес `http://127.0.0.1:8080/v1`), модели `ornithi`, `ornithit`, `qwen36i` с вариантами `fast` (`reasoningEffort: none`), `low`, `medium`, `high` (отправляет `xhigh`), `max`; `whitelist` показывает только эти три модели, `enabled_providers: ["tiel-local"]`, подсказка агента `agent.build.prompt = {file:~/.config/shuracode/prompt.txt}`, плагин `~/.local/share/shura/shuracode/plugin/server.ts`, MCP-серверы `memory`, `playwright`, `searxng` (через фильтр `mcp/filter.js` и списки `*-allow.json`). Модель `qwen36i` у Жомарта не заработает (нет её файла): сразу после шага 5 убери её из `whitelist` командой в 5.4.
**Обёртка** `~/.local/share/shura/shuracode/bin/shuracode` (исходная версия из репозитория + 9 наших строк): `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=65536` (лимит вывода за шаг; без него длинные записи файлов на режиме max обрываются), `TMPDIR=~/.cache/shuracode/tmp` (временные файлы движка не копятся в оперативной `/tmp`).

## 5. Шлюз, конфиги, ShuraCode, поиск

### 5.1. Репозитории
```bash
cd ~ && git clone --branch laptop https://github.com/hikkian/shura.git shura      # шлюз, скрипты, патчи, setup-bundle
git -C ~/shura log -1 --format='%h %s'
mkdir -p ~/.local/share/shura && git clone https://github.com/hikkian/shuracode.git ~/.local/share/shura/shuracode
git -C ~/.local/share/shura/shuracode rev-parse HEAD       # у нас a938815ea3cc95939f2ec4cd432f1b109986f374
```
[ПРОВЕРЕНО: оба репозитория публичны и отвечают (`git ls-remote`).] Репозиторий `hikkian/shura` с веткой `laptop` содержит и свою публичную историю (с `main`), и каталог `laptop/`.
Каталог `~/.local/share/shura/shura.cpp` создан на шаге 2.

### 5.2. Установка ShuraCode (движок OpenCode 1.18.32) [НЕ ПРОВЕРЕНО на чистой системе; у нас так установлен]
```bash
cd ~/.local/share/shura/shuracode && ./install.sh
~/.local/bin/shuracode --version        # ShuraCode 0.1.0 (engine: OpenCode 1.18.32)
```
`install.sh` ставит `npm install opencode-ai@1.18.32` (около 190 МБ), запускает самопроверку `tests/selftest.py` и создаёт `~/.local/bin/shuracode`. Если `~/.local/bin` не в `PATH`, добавь в `~/.bashrc`: `export PATH=$HOME/.local/bin:$PATH`.
Пакеты плагинов для конфигурации: `cd ~/.config/shuracode && npm install` после шага 5.3 (в `package.json` `@opencode-ai/plugin 1.18.32`; пакет `@opencode-ai/sdk 1.18.32` подтянется как зависимость) [НЕ ПРОВЕРЕНО: у нас эти пакеты оказались в каталоге иным путём].

### 5.3. Наши ручные правки поверх репозиториев: `apply-bundle.sh`
Что внутри `laptop/setup-bundle/` (все файлы без секретов):
- `gateway/`: `ai_gateway.py` (наши правки: языковая защита, защита от опечаток, потолки мыслей на шаг, эндпоинты управления), `lang_guard.py`, `typo_fix.py`, `game_mode.py`, `gpu_nvml.py`. **Без `lang_guard.py` и `typo_fix.py` шлюз не стартует** (импорт в строках 34-35). В репозитории `shura` их нет.
- `config/`: `model-launch.json`, `guardian.json` (в `.gitignore` репозитория, в git их нет), `templates/ornith-ideal5.jinja` и `qwen36-ideal5.jinja` (шаблон чата ideal5), `moe-trace/*.csv`, `shuracode.json`, `AGENTS.md`, `prompt.txt` (подсказка агента **v6**, sha256 `cfa8dccf0d011af5314a5cc6f593cd524de8a785635126e4147e52c6d78719e2`), `package.json`, `commands/`, `mcp/` (фильтры инструментов).
- `shuracode/`: `shuracode-wrapper` (обёртка), `server.ts` (плагин: переименование, `shell.env` с безопасным `pkill`, автоматическое чтение текста с картинок через `tesseract`), `safe-bin/` (`pkill` и `pgrep` с `--ignore-ancestors`: без них команда остановки процесса убивает собственную оболочку агента и вешает его на 2 минуты), `shura-launcher` (одна строка отличается от `scripts/shura`).
- `systemd/` юниты шлюза и поиска, `searxng/settings.yml`, `system/cpu-cap.sh`, `engine/` (патч движка и коммит).
```bash
bash ~/shura/laptop/setup-bundle/apply-bundle.sh
```
Скрипт делает резервные копии заменяемых файлов в `~/.shura-setup-backup/<время>/`, заменяет `/home/hikkian` на `$HOME` в текстовых конфигах и юнитах. [ПРОВЕРЕНО: запускали на пустом домашнем каталоге; все JSON валидны, строк `/home/hikkian` в результате 0.]
Системный Python-шлюзу библиотеки не нужны (только стандартная библиотека).

### 5.4. Убрать из списка модели, которых у Жомарта нет [НЕ ПРОВЕРЕНО на его машине; на нашей аналог выполнен]
```bash
python3 - <<'E'
import json,os
p=os.path.expanduser('~/.config/shuracode/shuracode.json'); c=json.load(open(p))
c['provider']['tiel-local']['whitelist']=['ornithi','ornithit']
json.dump(c,open(p,'w'),indent=2,ensure_ascii=False)
E
```

### 5.5. Поиск SearXNG (контейнер)
```bash
mkdir -p ~/searxng-config
# apply-bundle.sh настройки поиска не кладёт (нужен секрет); сгенерируй секрет и запиши файл:
sed "s#<GENERATE: openssl rand -hex 32>#$(openssl rand -hex 32)#" ~/shura/laptop/setup-bundle/searxng/settings.yml > ~/searxng-config/settings.yml
podman run -d --name searxng -p 127.0.0.1:8888:8080 -v ~/searxng-config:/etc/searxng:Z docker.io/searxng/searxng:latest
sleep 15; curl -s "http://127.0.0.1:8888/search?q=Алматы+новости&format=json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['results']),'результатов')"
```
Ожидается: 20-30 результатов. [ПРОВЕРЕНО у нас: в настройках отключены DuckDuckGo, Brave, Startpage (отвечают капчей) и включены Bing News, Bing, mwmbl, GitHub, StackOverflow; на наши запросы 16-125 результатов. Команда `podman run` в такой форме [НЕ ПРОВЕРЕНА]: у нас контейнер создан вручную с портом `0.0.0.0:8888` и монтированием `~/searxng-config`; здесь порт ограничен адресом 127.0.0.1.] Если контейнер выдаёт ошибку прав на `settings.yml`, сделай файл читаемым для всех: `chmod 644 ~/searxng-config/settings.yml`.
Юнит `searxng.service` запускает `podman start searxng` при входе: `systemctl --user daemon-reload && systemctl --user enable searxng.service`.

### 5.6. Запуск шлюза
```bash
systemctl --user daemon-reload
systemctl --user enable --now ai-gateway.service
sleep 5; curl -s localhost:8080/health
```
Ожидается: `{"status": "ok", "gateway": "ok", "model": "", "model_status": "UNLOADED"}` (модель не грузится, пока не придёт первый запрос). Журнал: `journalctl --user -u ai-gateway -f`. [ПРОВЕРЕНО у нас: этот ответ, юнит с `Environment=AI_GATEWAY_CONFIG=~/shura/config`, `LD_LIBRARY_PATH` движка из `ai-gateway.service.d/engine.conf`.]
**Если юнит не стартует:** `journalctl --user -u ai-gateway -n 50`; типичные причины: нет `lang_guard.py`/`typo_fix.py` (`ModuleNotFoundError`), нет файла `~/shura/config/model-launch.json` или в нём ссылка на несуществующий путь.
**Включить и выключить:** `curl -s -X POST localhost:8080/guardian/unload` (выгрузить модель; [ПРОВЕРЕНО]), `-X POST .../guardian/ai-off` (запретить загрузку, пока не включат), `-X POST .../guardian/ai-on` (разрешить), `curl localhost:8080/guardian/status` (состояние, память). Остановить всё: `systemctl --user stop ai-gateway` (шлюз сам сохранит сессию и остановит llama-server).
**Откат шага 5:** `systemctl --user disable --now ai-gateway searxng; podman rm -f searxng`; файлы вернуть из `~/.shura-setup-backup/<время>/`.

## 6. Шаблон ideal5 и подсказка v6

- Шаблон чата: `~/shura/config/templates/ornith-ideal5.jinja` (шаблон Qwen3.6 с нашими правками: одна фиксированная фраза о языке, уровень усилия одной последней строкой в system, кэш префикса не ломается: 0 разрывов из 32 ходов). Подключён в `model-launch.json` через `--chat-template-file` (`extraArgs`).
- Подсказка агента: `~/.config/shuracode/prompt.txt` (v6 = v4 + два правила: «сохрани первую рабочую версию рано» и «выпиши все названные границы и проверь каждую»), sha256 `cfa8dccf0d011af5314a5cc6f593cd524de8a785635126e4147e52c6d78719e2`; подключена в `shuracode.json` как `agent.build.prompt = {file:~/.config/shuracode/prompt.txt}`.
Проверка: `sha256sum ~/.config/shuracode/prompt.txt ~/shura/laptop/setup-bundle/config/prompt.txt` (одинаковые) и `grep -c "Save a first complete working version" ~/.config/shuracode/prompt.txt` (1).
**Откат подсказки:** старые версии лежат в `~/shura/laptop/ideal5/prompt-v4.txt`.

## 7. ShuraCode: плагины и запуск

Плагин `server.ts` (в бандле) делает три вещи: переименовывает движок в системном тексте, подставляет безопасные `pkill`/`pgrep` первыми в PATH для команд агента, и при вложенной картинке запускает `magick` (увеличение вдвое) и `tesseract -l rus+eng --psm 6` и добавляет распознанный текст к сообщению. Без `tesseract`/`ImageMagick` плагин молча ничего не меняет.
Запуск: `cd ~/любой-проект && shuracode`. Одноразовый запрос: `shuracode run --variant medium "задача" < /dev/null`. **Всегда добавляй `< /dev/null` в неинтерактивных запусках**, иначе `shuracode run` зависает, ожидая stdin.

---

## 8. Проверки после установки

Выполняй по порядку. Критерии пройдены, только если все пункты зелёные.

### 8.1. Модель грузится, скорость, память и видеопамять [ПРОВЕРЕНО у нас, значения ниже наши]
```bash
cd ~/shura/laptop/tools/weights-fix && python3 measure_comfort.py ornithi
```
Скрипт перезапускает шлюз, грузит модель, делает три генерации по 450 токенов и печатает JSON. Что должно получиться на таком же ноутбуке (у нас: `ornithi` с `l39p`):
- `load_s` 11-14 (секунд на загрузку);
- `tok_s`: три числа **36-39** токенов/с (короткий контекст, мышление выключено, без вложений);
- `avail_after_gen_mb`: **4300-5600** (свободно ОЗУ после генерации; зависит от открытых программ, у нас без браузера 5,5 ГБ, с браузером и VS Code 4,3 ГБ). Меньше **3000 МБ** значит, что что-то лишнее открыто или модель не та;
- `vram_mb` около **5750-5770** (из 6144);
- `rss_after_gen_mb` около **12400-12700**.
Скорость на длинном контексте: по нашим прежним замерам около 42 ток/с на 30 тыс. контекста и 36 ток/с при чтении 139 тыс. токенов [из отчёта `REPORT-home-pc.md` §4, повторно не замеряли].

### 8.2. Свободная память и видеопамять вручную
```bash
free -m | sed -n 2p                                  # столбец available: см. 8.1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader     # ~5760 MiB, 6144 MiB
```

### 8.3. Русский ответ без опечаток и без чужих букв [ПРОВЕРЕНО: команды выполнены дословно; словарь счёл неизвестными только «кэша», «кэше», «кэширования»]
```bash
mkdir -p /tmp/shura-test && cd /tmp/shura-test && git init -q .
shuracode run --variant high --format json "Объясни по-русски в четырёх предложениях, как работает кэш процессора и зачем нужен мониторинг памяти. Ничего не создавай." < /dev/null > out.jsonl
python3 - <<'PYEOF'
import json
t = ''
for l in open('/tmp/shura-test/out.jsonl'):
    try: e = json.loads(l)
    except Exception: continue
    if e.get('type') == 'text': t += e['part'].get('text', '')
open('/tmp/shura-test/answer.txt', 'w').write(t); print(t)
PYEOF
grep -oE '[А-Яа-яЁё]{4,}' /tmp/shura-test/answer.txt | sort -u | hunspell -d ru_RU -l
```
Проверка орфографии (последняя команда выше) должна вывести только технический жаргон (токен, промпт, кэш...), а не искажённые слова. Ответ целиком по-русски: латиницы в прозе меньше 30%. [ПРОВЕРЕНО у нас: ответ на русском, неизвестны словарю были только «токенов» и «промпта».]
Финальный русский ответ появляется целиком после завершения (потока нет): так работает проверка опечаток.

### 8.4. Вызов инструментов (чтение, правка, shell) [ПРОВЕРЕНО: команды этого пункта выполнены дословно, `calc.py` исправлен, вывод `5`]
```bash
mkdir -p /tmp/shura-tools && cd /tmp/shura-tools && git init -q . && printf 'def add(a, b):\n    return a - b\n' > calc.py
shuracode run --variant medium "В файле calc.py функция add считает неправильно. Исправь её, потом запусти python3 -c 'from calc import add; print(add(2,3))' и покажи результат." < /dev/null | tail -20
cat calc.py; python3 -c 'from calc import add; print(add(2,3))'
```
Ожидается: в `calc.py` стоит `a + b`, команда печатает `5`. Время 1-3 минуты. Если агент «зависает» минут на 2 на команде остановки процесса, проверь `safe-bin` (шаг 5.3): `command -v pkill` внутри команд агента должен вести в `~/.local/share/shura/shuracode/safe-bin`.

### 8.5. Одна задача из `laptop/bench`
Берём задачу h2 (JS-пул): `~/shura/laptop/heldout/h2_js_pool/` (в каталоге `prompt.txt` и `repo/`).
```bash
rm -rf /tmp/h2 && mkdir /tmp/h2 && cp -r ~/shura/laptop/heldout/h2_js_pool/repo/. /tmp/h2/ && cd /tmp/h2 && git init -q . && git add -A && git commit -qm init
shuracode run --variant high "$(cat ~/shura/laptop/heldout/h2_js_pool/prompt.txt)" < /dev/null | tail -15
node --test 2>&1 | tail -8
```
Ожидается: видимые тесты проходят (`node --test` без падений). У нас на этой задаче 3 из 3 прогонов проходили за 3-8 минут. Скрытые тесты (`hidden_test.py`) запускать не обязательно. Если задача не решилась с первого раза, повтори один раз: результат модели колеблется.

### 8.6. Поиск и картинка (по желанию)
- Поиск: попроси `shuracode`: «Найди свежие новости про Алматы и перечисли заголовки со ссылками»: должны прийти результаты из `searxng`.
- Картинка: `shuracode run --variant high "Какой номер БИН внизу?" -f ~/shura/laptop/heldout/h4_invoice_mockup/attach.png < /dev/null` (файл макета: `laptop/heldout/h4_invoice_mockup/attach.png`, если каталог есть). Ожидается ответ `123456789012`. [ПРОВЕРЕНО у нас: верно, за 68 секунд.] Требуются `tesseract` и `ImageMagick` из шага 1.4.

---

## 9. Чего не делать и отличия от нашего домашнего ПК

**У нас на домашнем ПК (RX 7800 XT, ROCm) другое железо и сборка.** Для Жомарта (NVIDIA, CUDA) НЕ применимо:
- ROCm, `amdgpu`, сборка движка под HIP, замеры скорости ПК (75 ток/с у нас относятся к другой модели и другой видеокарте), планы «Q4_K_M на ПК», VRAM-governor через `amdgpu sysfs` (по умолчанию выключен, `vramGuard false`);
- KDE-скрипты `gpu-mode.sh`, `gpu-by-power.sh`, `~/.config/plasma-workspace/env/` (на GNOME не нужны; у нас они даже не установлены), `kscreen`;
- игровой режим шлюза (`gameMode`, он по умолчанию выключен, оставь `false`; список процессов в `guardian.json` содержит `kwin_wayland`, `plasmashell`, на GNOME он ни на что не влияет).

**Что относится к нему так же, как к нам:**
- **Сторож памяти шлюза** (`guardian.json`): модель не грузится при менее 14 ГБ свободной ОЗУ и выгружается при простое 15 минут; не повышай `idleUnloadSeconds` и не отключай защиту без причины. Правило «свободно не меньше 4 ГБ при работе агента» держи в голове: не открывай тяжёлые программы одновременно с моделью (браузер около 1,7 ГБ, VS Code 0,8 ГБ).
- **Не трогать боевые порты 8080 (шлюз) и 8090 (llama-server)**, пока кто-то работает с агентом. Для опытов с моделью запускай отдельный `llama-server` на порту **8092** (как в `laptop/tools/weights-fix/eval_variant.sh`) и только когда рабочий остановлен: `systemctl --user stop ai-gateway` (ОЗУ не хватит на две копии модели).
- Зрительный блок (`mmproj`) должен оставаться в памяти (`alwaysVision: true`), иначе первое сообщение с картинкой ждёт перезагрузку модели.
- Не запускай тяжёлые прогоны бенча вместе с рабочим агентом.
- `pkill -f "..."` с текстом из той же команды убивает собственную оболочку; в скриптах убивай по PID.
- **Системные настройки нашего ноутбука, которые делают его комфортным (по желанию, [НЕ ПРОВЕРЕНО на его машине]):**
  - zram: `/etc/systemd/zram-generator.conf` (zstd, размер равен ОЗУ) и `/etc/sysctl.d/99-zram.conf` с `vm.swappiness = 180`: даёт запас памяти за счёт сжатого свопа. Содержимое смотри в `~/shura/laptop/setup-bundle/system/` (если файлов нет, спроси автора);
  - ограничение мощности процессора `sudo ~/shura/laptop/setup-bundle/system/cpu-cap.sh 45 45` (пределы PL1 и PL2 по 45 Вт; без него процессор доходил до 97 °C на долгих задачах, с ним 65-75 °C). Не сохраняется после перезагрузки; вернуть заводские: `sudo cpu-cap.sh reset`;
  - андервольт процессора `/etc/intel-undervolt.conf` и настройки BIOS у нас сделаны вручную и зависят от конкретного экземпляра ноутбука: **не копируй** (`intel-undervolt` без проверки может привести к зависаниям).

## 10. Откат по шагам (сводка)

| Шаг | Откат |
|---|---|
| 1 драйвер и CUDA | `sudo dnf remove 'cuda-*' akmod-nvidia 'xorg-x11-drv-nvidia*' 'kmod-nvidia*'; sudo reboot` |
| 2 движок | `rm -rf ~/.local/share/shura/shura.cpp/build-cuda`; `git -C ~/.local/share/shura/shura.cpp checkout -- . && git -C ... clean -fd`; `podman rmi docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04` |
| 3 модель | удалить файлы из `~/.local/share/shura/models/` и `~/shura-work` |
| 3 откат к версии без слоя 39 | в `~/shura/config/model-launch.json` для `ornithi` и `ornithit` поставить `modelPath` на `...-3bit-MTP.gguf`, `systemctl --user restart ai-gateway` |
| 4-6 конфиги и подсказка | `cp -a ~/.shura-setup-backup/<время>/. ~/` (резервные копии скрипта `apply-bundle.sh`), затем `systemctl --user daemon-reload && systemctl --user restart ai-gateway` |
| 5 шлюз и поиск | `systemctl --user disable --now ai-gateway searxng; podman rm -f searxng` |
| 7 ShuraCode | `rm -rf ~/.local/share/shura/shuracode ~/.local/share/shuracode ~/.local/bin/shuracode ~/.config/shuracode` |

Наши рабочие откаты у автора (для справки): `apply-production/rollback.sh`, `weights-fix/rollback-l39p.sh`; в репозиторий они попали как `laptop/tools/apply-production/` и `laptop/tools/weights-fix/rollback-l39p.sh`.

## 11. Шпаргалка на каждый день

```bash
# включить (после перезагрузки всё стартует само, если выполнены enable в шаге 5):
systemctl --user start searxng ai-gateway
curl -s localhost:8080/health                      # {"status": "ok", ...}

# работать:
cd ~/проект && shuracode                           # откроется ShuraCode; первый запрос грузит модель около 15 секунд
# режимы усилия: fast, low, medium, high, max (переключение клавишей Ctrl+T или shuracode run --variant high "...")
# medium для обычного, high для сложного, max очень долгий

# выключить модель, освободить 11 ГБ ОЗУ и видеопамять:
curl -s -X POST localhost:8080/guardian/unload     # {"ok": true}
# совсем остановить:
systemctl --user stop ai-gateway searxng

# посмотреть состояние и память:
curl -s localhost:8080/guardian/status | head -c 400; free -m | sed -n 2p
```
Если что-то пошло не так: `journalctl --user -u ai-gateway -n 50`, `shuracode doctor`, затем раздел 10.

---

## Приложение A. Расхождения и ручные правки, о которых надо знать

- Установщик `setup.sh` репозитория `hikkian/shura` собирает другой движок и кладёт устаревший шаблон `config/shuracode.json.in` (модель `tiel-coder`, без `whitelist` и без подсказки): **его не запускай**, используй шаги этой инструкции.
- В `ARCHITECTURE.md` написано «выгрузка при простое 30 минут»; реально стоит `idleUnloadSeconds = 900` (15 минут) в `guardian.json`.
- Сервис поиска у нас слушал `*:8888` (все интерфейсы) и монтировал `~/searxng-config`; в инструкции порт привязан к `127.0.0.1`.
- Каталог `~/.config/shuracode/node_modules` у нас появился не через `install.sh`; шаг 5.2 даёт `npm install`, как это сделать, мы не записывали.
- В `shura.cpp` 7 файлов правятся патчем; патч в самом репозитории `shura.cpp` не закоммичен: используй `setup-bundle/engine/shura.cpp-uncommitted.patch`.
- Системные файлы вне репозиториев (zram, sysctl, intel-undervolt, BIOS): шаг 9, копировать только zram и `cpu-cap.sh` по желанию.

## Приложение B. Что подтверждено, а что нет

**Проверено у нас (запускали, результат совпал):** версии драйвера, CUDA, пакетов; sha256 и размеры всех файлов моделей; ссылки Hugging Face; сборка головы MTP, базы с MTP и итогового `l39p` с побитовым совпадением; донор Q4_K `--pure`; патч движка применяется к чистому коммиту; контейнер сборки настраивается и компилирует первую цель; скрипт `apply-bundle.sh` на пустом домашнем каталоге; `/health`, `/guardian/unload`, `/guardian/status` шлюза; скорость, память и видеопамять модели `l39p`; русский ответ и вызов инструментов; поиск; чтение текста с картинки.
**Не проверено:** полная сборка движка (долгая); установка драйвера и CUDA на чистой Fedora (команды по документации, пакеты и версии у нас подтверждены); чистая установка шлюза и ShuraCode на новой машине с нуля; всё про GNOME; создание контейнера поиска именно такой командой; zram и `cpu-cap.sh` на его ноутбуке; побитовое совпадение донора при другой версии `llama-quantize` (поэтому в 3.7 есть проверка по трём тензорам).
