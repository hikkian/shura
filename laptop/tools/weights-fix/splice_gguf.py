#!/home/hikkian/Desktop/AI/bench/.venv/bin/python
"""Подмена тензоров базового GGUF тензорами из GGUF-донора (другой квант).

Пример:
  splice_gguf.py --base B.gguf --donor D.gguf --out O.gguf \
      --layers 3,7,12-15 --kinds ffn_gate_exps,ffn_up_exps,ffn_down_exps [--dry-run]

Все KV и остальные тензоры копируются из базы без изменений. Запись потоковая
(mmap -> файл кусками), пишется O.tmp, затем rename. Слой MTP (содержит
blk.L.nextn.*) подменяется только если указан явно одиночным числом (не в диапазоне).
"""
import argparse, hashlib, os, random, re, sys
from pathlib import Path

import numpy as np
# gguf-py: сначала из shura.cpp; если он не импортируется (сейчас SyntaxError в tensor_mapping.py) — запасные копии
for _p in (os.environ.get("GGUF_PY"), "/home/hikkian/.local/share/shura/shura.cpp/gguf-py",
           "/home/hikkian/.unsloth/llama.cpp/gguf-py", "/home/hikkian/llama-cpp-turboquant/gguf-py"):
    if not _p or not os.path.isdir(_p):
        continue
    for _m in [m for m in sys.modules if m == "gguf" or m.startswith("gguf.")]:
        del sys.modules[_m]
    sys.path.insert(0, _p)
    try:
        import gguf
        print(f"[gguf-py: {_p}]", file=sys.stderr)
        break
    except (SyntaxError, ImportError) as _e:
        print(f"[gguf-py {_p} не импортируется: {_e.__class__.__name__}]", file=sys.stderr)
        sys.path.pop(0)
else:
    raise SystemExit("не найден рабочий gguf-py")
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import quant_shape_to_byte_shape

CHUNK = 64 << 20  # 64 МиБ


def parse_layers(s):
    """-> (set слоёв, set явно указанных одиночных)"""
    allv, single = set(), set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m[1]), int(m[2])
            if a > b:
                raise SystemExit(f"плохой диапазон {part}")
            allv.update(range(a, b + 1))
        elif part.isdigit():
            allv.add(int(part)); single.add(int(part))
        else:
            raise SystemExit(f"плохой элемент --layers: {part!r}")
    return allv, single


def expected_nbytes(t):
    """Размер по ggml-формуле + проверка делимости строки на блок."""
    bs, ts = GGML_QUANT_SIZES[t.tensor_type]
    ne0 = int(t.shape[0])
    if ne0 % bs:
        raise ValueError(f"{t.name}: ne0={ne0} не кратно блоку {bs} ({t.tensor_type.name})")
    n = (ne0 // bs) * ts
    for d in t.shape[1:]:
        n *= int(d)
    return n


def check_tensor_size(t, reader, gaps):
    exp = expected_nbytes(t)
    if exp != t.n_bytes:
        raise ValueError(f"{t.name}: формула {exp} != reader {t.n_bytes}")
    if t.tensor_type not in (QT.F32, QT.F16, QT.F64, QT.I8, QT.I16, QT.I32, QT.I64):
        bshape = quant_shape_to_byte_shape(tuple(reversed([int(x) for x in t.shape])), t.tensor_type)
        if int(np.prod(bshape, dtype=np.uint64)) != exp:
            raise ValueError(f"{t.name}: byte-shape {bshape} != {exp}")
    gap = gaps[t.name]  # расстояние до следующего тензора (или до EOF)
    if not (exp <= gap < exp + reader.alignment):
        raise ValueError(f"{t.name}: размер {exp} не вписывается в зазор {gap} (align {reader.alignment})")
    return exp


def tensor_gaps(reader, file_size):
    ts = sorted(reader.tensors, key=lambda t: t.data_offset)
    gaps = {}
    for i, t in enumerate(ts):
        end = ts[i + 1].data_offset if i + 1 < len(ts) else None
        if end is None:
            # последний: допускаем отсутствие хвостового паддинга
            gaps[t.name] = t.n_bytes if file_size - t.data_offset >= t.n_bytes else -1
            if file_size - t.data_offset < t.n_bytes:
                raise ValueError(f"{t.name}: файл обрезан")
        else:
            gaps[t.name] = end - t.data_offset
    return gaps


def raw(reader, t):
    return reader.data[t.data_offset:t.data_offset + t.n_bytes]


def sha(reader, t):
    h = hashlib.sha256()
    off, end = t.data_offset, t.data_offset + t.n_bytes
    while off < end:
        n = min(CHUNK, end - off)
        h.update(memoryview(reader.data[off:off + n]))
        off += n
    return h.hexdigest()


def fmt_gb(x):
    return f"{x / 2**30:+.3f} ГиБ"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--donor", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layers", default="")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--names", default="", help="полные имена тензоров вне слоёв, напр. token_embd.weight,output.weight")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="seed выбора тензоров для самопроверки")
    a = ap.parse_args()

    layers, single = parse_layers(a.layers) if a.layers else (set(), set())
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    names = [x.strip() for x in a.names.split(",") if x.strip()]
    if not names and (not layers or not kinds):
        raise SystemExit("пустые --layers/--kinds (или задайте --names)")

    base = GGUFReader(a.base, "r")
    donor = GGUFReader(a.donor, "r")
    bmap = {t.name: t for t in base.tensors}
    dmap = {t.name: t for t in donor.tensors}

    # MTP-слои: те, где есть blk.L.nextn.*
    mtp = set()
    for n in bmap:
        m = re.match(r"blk\.(\d+)\.nextn\.", n)
        if m:
            mtp.add(int(m[1]))
    for L in sorted(layers & mtp - single):
        print(f"ПРЕДУПРЕЖДЕНИЕ: слой {L} — MTP, в диапазоне он пропущен (укажите его отдельным числом).")
    layers = {L for L in layers if L not in mtp or L in single}
    for L in sorted(layers & mtp):
        print(f"ВНИМАНИЕ: явно подменяется MTP-слой {L}")

    bgaps = tensor_gaps(base, a.base.stat().st_size)
    dgaps = tensor_gaps(donor, a.donor.stat().st_size)

    plan = []  # (layer, kind, name)
    errors = []
    for L in sorted(layers):
        for k in kinds:
            name = f"blk.{L}.{k}.weight"
            if name not in bmap: errors.append(f"нет в базе: {name}"); continue
            if name not in dmap: errors.append(f"нет в доноре: {name}"); continue
            bt, dt = bmap[name], dmap[name]
            if not np.array_equal(bt.shape, dt.shape):
                errors.append(f"{name}: shape база {bt.shape.tolist()} != донор {dt.shape.tolist()}"); continue
            if bt.tensor_type == dt.tensor_type:
                errors.append(f"{name}: тип одинаковый ({bt.tensor_type.name})"); continue
            try:
                check_tensor_size(dt, donor, dgaps)
                check_tensor_size(bt, base, bgaps)
            except ValueError as e:
                errors.append(str(e)); continue
            plan.append((L, k, name))
    for name in names:
        if name not in bmap or name not in dmap: errors.append(f"нет в базе/доноре: {name}"); continue
        bt, dt = bmap[name], dmap[name]
        if not np.array_equal(bt.shape, dt.shape): errors.append(f"{name}: shape различается"); continue
        if bt.tensor_type == dt.tensor_type: errors.append(f"{name}: тип одинаковый"); continue
        try:
            check_tensor_size(dt, donor, dgaps); check_tensor_size(bt, base, bgaps)
        except ValueError as e:
            errors.append(str(e)); continue
        plan.append((-1, name, name))
    if errors:
        print("ОШИБКИ проверки:", *errors, sep="\n  ")
        raise SystemExit(2)
    if not plan:
        raise SystemExit("нечего подменять")

    total = 0
    print(f"{'слой':>4} {'вид':<14} {'база':>6} -> {'донор':<6} {'Δ байт':>14}")
    for L, k, n in plan:
        d = dmap[n].n_bytes - bmap[n].n_bytes
        total += d
        print(f"{L:>4} {k:<14} {bmap[n].tensor_type.name:>6} -> {dmap[n].tensor_type.name:<6} {d:>+14,}")
    print(f"Итого подмен: {len(plan)}, прирост {total:,} байт ({fmt_gb(total)}); "
          f"размер базы {a.base.stat().st_size / 2**30:.3f} ГиБ, ожидаемый результат ≈ "
          f"{(a.base.stat().st_size + total) / 2**30:.3f} ГиБ")
    print("ПРЕДУПРЕЖДЕНИЕ: general.file_type и прочие KV оставлены как в базе (не отражают смешанный квант).")
    if a.dry_run:
        print("--dry-run: запись не выполнялась.")
        return

    # ---- запись ----
    repl = {n for _, _, n in plan}
    arch = base.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = GGUFWriter(None, arch, endianess=gguf.GGUFEndian.LITTLE)  # путь задаём ниже
    for field in base.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    writer.data_alignment = base.alignment  # то же выравнивание, что и в базе

    srcs = []  # (reader, tensor) в порядке базы
    for bt in base.tensors:
        if bt.name in repl:
            r, t = donor, dmap[bt.name]
        else:
            r, t = base, bt
        srcs.append((r, t))
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)

    tmp = Path(str(a.out) + ".tmp")
    writer.path = tmp
    try:
        writer.open_output_file(tmp)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        fo = writer.fout[0]
        writer.write_padding(fo, fo.tell())
        done = 0
        for r, t in srcs:
            off, end = t.data_offset, t.data_offset + t.n_bytes
            while off < end:
                n = min(CHUNK, end - off)
                fo.write(memoryview(r.data[off:off + n]))
                off += n
            writer.write_padding(fo, t.n_bytes)
            done += 1
            if done % 100 == 0:
                print(f"  записано {done}/{len(srcs)}", flush=True)
        writer.close()
    except BaseException:
        try: writer.close()
        except Exception: pass
        raise
    os.replace(tmp, a.out)
    print(f"Записано: {a.out} ({a.out.stat().st_size / 2**30:.3f} ГиБ)")

    # ---- самопроверка ----
    verify(a.out, base, donor, bmap, dmap, repl, a.seed)


def verify(out, base, donor, bmap, dmap, repl, seed):
    rnd = random.Random(seed)
    res = GGUFReader(out, "r")
    ok = True
    def bad(msg):
        nonlocal ok; ok = False; print("ПРОВАЛ:", msg)
    if res.alignment != base.alignment: bad("alignment")
    if [t.name for t in res.tensors] != [t.name for t in base.tensors]:
        bad("список/порядок тензоров")
    # KV
    bk = [f for f in base.fields if not f.startswith("GGUF.")]
    rk = [f for f in res.fields if not f.startswith("GGUF.")]
    if sorted(bk) != sorted(rk): bad("набор ключей KV")
    for k in bk:
        if k in res.fields:
            v1, v2 = base.fields[k].contents(), res.fields[k].contents()
            if v1 != v2 or base.fields[k].types != res.fields[k].types:
                bad(f"KV {k} отличается")
    rmap = {t.name: t for t in res.tensors}
    for n in repl:
        if rmap[n].tensor_type != dmap[n].tensor_type: bad(f"тип {n}")
        if not np.array_equal(rmap[n].shape, dmap[n].shape): bad(f"shape {n}")
    for n, t in rmap.items():
        if n not in repl and (t.tensor_type != bmap[n].tensor_type or not np.array_equal(t.shape, bmap[n].shape)):
            bad(f"тип/shape неподменённого {n}")
    others = [n for n in rmap if n not in repl]
    pick = rnd.sample(others, min(3, len(others)))
    pr = rnd.choice(sorted(repl))
    for n in pick:
        h1, h2 = sha(res, rmap[n]), sha(base, bmap[n])
        print(f"  sha256 {n}: {'OK' if h1 == h2 else 'РАЗЛИЧАЕТСЯ'} {h1[:16]}")
        if h1 != h2: ok = False
    h1, h2 = sha(res, rmap[pr]), sha(donor, dmap[pr])
    print(f"  sha256 {pr} (подменён, = донор): {'OK' if h1 == h2 else 'РАЗЛИЧАЕТСЯ'} {h1[:16]}")
    if h1 != h2: ok = False
    print("САМОПРОВЕРКА:", "ПРОЙДЕНА" if ok else "ПРОВАЛЕНА")
    if not ok:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
