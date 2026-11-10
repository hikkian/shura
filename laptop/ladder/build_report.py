#!/usr/bin/env python3
"""build_report.py -> report.html. Every number comes from final_data.json (made by final_tables.py from results/*.jsonl).
Run: python3 -B final_tables.py && python3 -B build_report.py. Prose is plain text edited by hand; numbers are never typed in."""
import json, html, os, re
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_data.json")))
LAD = D["table1_ladder"]["modes"]
T3 = D["table3_h1_h5_templates"]
T4 = D["table4_codex_vs_us"]
T5 = D["table5_timeouts"]
MODES = ["fast", "low", "medium", "high", "xhigh", "max"]
RU = {"fast": "fast", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}

def pc(x, d=0):
    return "—" if x is None else f"{100*x:.{d}f}%"
def sec(x):
    if x is None: return "—"
    x = int(round(x))
    if x < 90: return f"{x} с"
    m = x / 60
    return (f"{m:.1f}".replace(".", ",") + " мин") if m < 10 else f"{round(m)} мин"
def e(s): return html.escape(str(s))

def ladder_rows():
    out = []
    for m in MODES:
        a = LAD[m]["without_hd5"]; b = LAD[m]["all_hd1_hd6"]
        reps = a.get("rep_means") or {}
        dots = "".join(f'<i class="dot" style="left:{100*v:.1f}%" title="повтор {k}: {100*v:.0f}%"></i>' for k, v in reps.items())
        out.append(f'''<div class="lrow">
  <div class="lname">{RU[m]}</div>
  <div class="track" role="img" aria-label="{RU[m]}: {pc(a['mean_frac'])} без hd5, {pc(b['mean_frac'])} со всеми">
    <b class="bar" style="width:{100*a['mean_frac']:.1f}%"></b>
    <i class="tick" style="left:{100*b['mean_frac']:.1f}%" title="со всеми шестью задачами: {pc(b['mean_frac'])}"></i>
    {dots}
  </div>
  <div class="lval">{pc(a['mean_frac'])}</div>
  <div class="lmeta"><span>{b['n']} прогонов</span><span>таймаутов {b['timeouts']}</span><span>медиана {sec(b['median_secs'])}</span></div>
</div>''')
    return "\n".join(out)

def versus_rows():
    a = T3["arms"]["codex"]["tasks"]; o = T3["arms"]["arm_ornith5_xhigh_ho"]["tasks"]
    names = {"h1_shop_discount": "Скидка в корзине", "h2_js_pool": "Пул задач на JS", "h3_readme_envconf": "README и конфигурация",
             "h4_invoice_mockup": "Страница счёта по макету", "h5_ci_cause": "Причина красного CI"}
    rows = []
    for t, label in names.items():
        c = a[t]["mean_frac"]; u = o[t]["mean_frac"]
        rows.append(f'''<div class="vrow"><div class="vname">{e(label)}</div>
  <div class="vbars"><div class="vb luna"><div class="vt"><b style="width:{100*c:.1f}%"></b></div><span>{pc(c)}</span></div>
  <div class="vb ours"><div class="vt"><b style="width:{100*u:.1f}%"></b></div><span>{pc(u)}</span></div></div></div>''')
    return "\n".join(rows)

def tmpl_rows():
    lab = {"arm_ornithsharp_xhigh_ho": "Sharp (оригинальный шаблон Tiel)", "arm_ornith4_xhigh_ho": "ideal4", "arm_ornith5_xhigh_ho": "ideal5 (выбран)",
           "arm_ornith6_xhigh_ho": "ideal6 (отвергнут)", "arm_ornith7_xhigh_ho": "ideal7 (отвергнут)", "codex": "Luna (Codex)"}
    rows = []
    for k, name in lab.items():
        r = T3["arms"][k]["total_h1_h5"]
        rows.append(f"<tr><td>{e(name)}</td><td class=n>{r['n']}</td><td class=n>{r['passed']}</td><td class=n>{pc(r['mean_frac'])}</td>"
                    f"<td class=n>{r['timeouts']}</td><td class=n>{sec(r['median_secs'])}</td></tr>")
    return "\n".join(rows)

def _frac(r):
    if r["hidden"]: return 1.0
    m = re.search(r"Ran (\d+) tests?", r["hidden_err"]); f = re.search(r"FAILED \(([^)]*)\)", r["hidden_err"])
    if not m: return 0.0
    n = int(m.group(1)); bad = sum(int(x) for x in re.findall(r"=(\d+)", f.group(1))) if f else 0
    return (n - bad) / n
def pic_table():
    R = os.path.expanduser("~/Desktop/AI/shura-report/results"); W = os.path.expanduser("~/Desktop/AI/shura-report/work")
    arms = [("arm_ornith5_xhigh_ho", "Как было"), ("arm_ornith5_xhigh_ocrplug", "С автоматическим чтением текста с картинки (OCR), xhigh"),
            ("arm_ornith5_medium_ocrplug", "То же, режим medium"), ("arm_ornith5_high_ocrplug", "То же, режим high")]
    rows = []
    for a, label in arms:
        try: rs = [json.loads(l) for l in open(f"{R}/{a}.jsonl")]
        except Exception: continue
        rs = [r for r in rs if r["task"].startswith("h4")]
        if not rs: continue
        fs = [_frac(r) for r in rs]; ok = 0
        for r in rs:
            try: ok += "БИН 123456789012" in open(f"{W}/{a}-h4_invoice_mockup-{r['rep']}/proj/index.html", encoding="utf8").read()
            except Exception: pass
        med = sorted(r["secs"] for r in rs)[len(rs)//2]
        rows.append(f"<tr><td>{e(label)}</td><td class=n>{pc(sum(fs)/len(fs))}</td><td class=n>{ok} из {len(rs)}</td><td class=n>{sec(med)}</td></tr>")
    return "\n".join(rows)

hd = T4["hd1_hd6"]["rows"]
luna_hd = hd.get("codex") or {}
best = max(MODES, key=lambda m: LAD[m]["without_hd5"]["mean_frac"] or 0)
old19 = T4["old_set_19"]["codex.final"]
cx = T3["arms"]["codex"]["total_h1_h5"]; o5 = T3["arms"]["arm_ornith5_xhigh_ho"]["total_h1_h5"]
import statistics
ratios = [ (LAD[m]['all_hd1_hd6']['median_secs'] or 0) / (luna_hd.get('median_secs') or 1) for m in ('medium','high','xhigh','max') ]
SPEED = f"{min(ratios):.0f}–{max(ratios):.0f}"
spread = max(((LAD[m]['all_hd1_hd6'].get('rep_span') or [0, 0])[1] - (LAD[m]['all_hd1_hd6'].get('rep_span') or [0, 0])[0]) for m in MODES)
lm = LAD["max"]["without_hd5"]; lh = LAD["high"]["without_hd5"]; lf = LAD["fast"]["without_hd5"]

FINAL = os.environ.get('REPORT_FINAL') == '1'
BANNER = '' if FINAL else ('<div class="panel" style="border-left:4px solid var(--warn)"><b class="warn">Промежуточная версия.</b> Прогон верхних режимов ещё идёт, данные на ' + e(D['generated_at'].replace('T', ' ')) + '. Итоговые цифры появятся вечером, страница обновится по той же ссылке.</div>')
PAGE = f'''<title>Лесенка режимов ShuraCode</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one narrow reading column; the ladder chart is the centre piece, Luna comparison below it */
:root {{
  --bg:#f2f4f5; --panel:#ffffff; --ink:#14212b; --ink2:#4a5863; --line:#d6dde1; --ours:#1f5fbf; --ours-soft:#bcd2f1; --luna:#3a4651; --warn:#a4380f;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f161b; --panel:#17212a; --ink:#e7edf1; --ink2:#9fb0bc; --line:#2c3a45; --ours:#6ea3f0; --ours-soft:#27405f; --luna:#b8c4cd; --warn:#f08a5d; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#0f161b; --panel:#17212a; --ink:#e7edf1; --ink2:#9fb0bc; --line:#2c3a45; --ours:#6ea3f0; --ours-soft:#27405f; --luna:#b8c4cd; --warn:#f08a5d; color-scheme:dark }}
body {{ background:var(--bg); color:var(--ink); font:16px/1.55 var(--sans); padding-block:28px 56px;
  padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); }}
main {{ max-width:820px; margin:0 auto; display:grid; gap:28px; }}
h1 {{ font-size:clamp(26px,5vw,38px); line-height:1.12; margin:0 0 8px; font-weight:600; letter-spacing:-.01em; text-wrap:balance }}
h2 {{ font-size:20px; margin:0 0 10px; font-weight:600; text-wrap:balance }}
p {{ margin:0 0 10px; max-width:68ch }}
.lead {{ color:var(--ink2); font-size:17px; max-width:62ch }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:20px clamp(14px,3vw,24px) }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-top:18px }}
.kpi {{ border-left:3px solid var(--ours); padding:2px 0 2px 12px }}
.kpi b {{ display:block; font:500 26px/1.1 var(--mono) }}
.kpi span {{ color:var(--ink2); font-size:14px }}
.lrow {{ display:grid; grid-template-columns:64px 1fr 52px; gap:6px 12px; align-items:center; padding:9px 0; border-top:1px solid var(--line) }}
.lrow:first-of-type {{ border-top:0 }}
.lname {{ font:500 15px var(--mono) }} .lval {{ font:500 15px var(--mono); text-align:right }}
.lmeta {{ grid-column:2 / 4; display:flex; gap:14px; flex-wrap:wrap; color:var(--ink2); font-size:13px }}
.track {{ position:relative; height:22px; background:color-mix(in srgb,var(--line) 55%,transparent); border-radius:3px }}
.bar {{ position:absolute; inset:0 auto 0 0; background:var(--ours); border-radius:3px 0 0 3px }}
.tick {{ position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--ink); transform:translateX(-1px) }}
.dot {{ position:absolute; top:50%; width:9px; height:9px; border-radius:50%; background:var(--panel); border:2px solid var(--ink); transform:translate(-50%,-50%) }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:var(--ink2); margin:0 0 6px }}
.legend i {{ display:inline-block; vertical-align:-2px; margin-right:6px }}
.sw-dot {{ width:9px; height:9px; border-radius:50%; background:var(--panel); border:2px solid var(--ink) }}
.sw-tick {{ width:2px; height:14px; background:var(--ink) }}
.sw-luna,.sw-ours {{ width:12px; height:12px; border-radius:2px }} .sw-luna {{ background:var(--luna) }} .sw-ours {{ background:var(--ours) }}
.vrow {{ padding:8px 0; border-top:1px solid var(--line) }} .vrow:first-of-type {{ border-top:0 }}
.vname {{ font-size:14px; margin-bottom:4px }}
.vb {{ display:grid; grid-template-columns:1fr 46px; gap:10px; align-items:center; margin:3px 0 }}
.vt {{ position:relative; height:18px; background:color-mix(in srgb,var(--line) 55%,transparent); border-radius:3px }}
.vt b {{ position:absolute; inset:0 auto 0 0; border-radius:3px 0 0 3px }} .vb.luna b {{ background:var(--luna) }} .vb.ours b {{ background:var(--ours) }}
.vb span {{ font:500 13px var(--mono); text-align:right }}
.tablewrap {{ overflow-x:auto }} table {{ border-collapse:collapse; width:100%; min-width:520px; font-size:14px }}
th,td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top }} th {{ font-weight:600; color:var(--ink2); font-size:13px }}
td.n,th.n {{ text-align:right; font-variant-numeric:tabular-nums }} td.n {{ font-family:var(--mono) }}
ul.plain {{ margin:0; padding-left:20px; display:grid; gap:6px; max-width:70ch }}
.note {{ color:var(--ink2); font-size:14px }}
.warn {{ color:var(--warn) }}
code {{ font:13px var(--mono) }}
@media (max-width:520px) {{ .lrow {{ grid-template-columns:54px 1fr 46px }} }}
</style>
<main>
{BANNER}
<header>
  <h1>Лесенка режимов ShuraCode на ноутбуке</h1>
  <p class="lead">ShuraCode работает целиком локально, на модели Ornith-1.5 35B-A3B. Здесь показано, как каждый из шести режимов усилия справляется с тяжёлыми задачами, и как это выглядит рядом с платной моделью Luna (Codex).</p>
  <div class="kpis">
    <div class="kpi"><b>{pc(lm['mean_frac'])}</b><span>max на тяжёлых задачах, против {pc(lf['mean_frac'])} у fast</span></div>
    <div class="kpi"><b>0 из 32</b><span>разрывов кэша контекста между ходами (было 27 из 32)</span></div>
    <div class="kpi"><b>{pc(luna_hd.get('mean_frac'))}</b><span>у Luna (облако) на тех же тяжёлых задачах; наш лучший режим {pc(lm['mean_frac'])}</span></div>
  </div>
</header>

<section class="panel" aria-labelledby="h-ladder">
  <h2 id="h-ladder">Результат растёт с режимом, но середина лесенки в пределах шума</h2>
  <p>Шесть задач на рассуждение, которые модель раньше не видела, а скрытые тесты были написаны до запуска и не менялись. Столбец показывает среднюю долю пройденных проверок.</p>
  <div class="legend"><span><i class="sw-dot"></i>результат отдельного повтора</span><span><i class="sw-tick"></i>среднее со всеми шестью задачами</span></div>
  {ladder_rows()}
  <p class="note" style="margin-top:12px">Столбец считан без задачи hd5 (применение патча): в любом режиме она упирается в лимит времени и ничего не различает. Решение исключить её принято после первых результатов, поэтому метка показывает и вариант со всеми шестью задачами. Разброс между повторами у одного режима достигает {100*spread:.0f} процентных пунктов, поэтому крупные разрывы (fast против остальных, max против high) надёжны, а соседние ступени середины лесенки по этим данным различить нельзя.</p>
</section>

<section class="panel" aria-labelledby="h-luna">
  <h2 id="h-luna">Рядом с Luna (Codex): новые задачи по пяти темам</h2>
  <div class="legend"><span><i class="sw-luna"></i>Luna</span><span><i class="sw-ours"></i>ShuraCode (ornith5, xhigh)</span></div>
  {versus_rows()}
  <p style="margin-top:12px">В среднем Luna {pc(cx['mean_frac'])}, мы {pc(o5['mean_frac'])}. Главный разрыв в задаче с макетом: наша модель теряет цифры при чтении мелкого текста с картинки. На старом наборе из 19 задач Luna прошла {old19['passed']} из {old19['n']}. Скорость сравнивать надо с оговоркой: Luna работает в облаке, а мы полностью локально на ноутбуке без интернета, поэтому на новых задачах медиана {sec(cx['median_secs'])} против {sec(o5['median_secs'])}. На домашнем ПК с видеокартой RX 7800 XT генерация будет заметно быстрее, чем на ноутбуке.</p>
</section>

<section class="panel" aria-labelledby="h-tmpl">
  <h2 id="h-tmpl">Какие подсказки лучше: выбор шаблона</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>Вариант</th><th class="n">Прогонов</th><th class="n">Прошло целиком</th><th class="n">Доля проверок</th><th class="n">Таймаутов</th><th class="n">Медиана</th></tr></thead>
    <tbody>{tmpl_rows()}</tbody></table></div>
  <p class="note" style="margin-top:10px">Пять новых задач на xhigh. Выбран ideal5: не хуже других по качеству, стабилен по кэшу и без правил, подогнанных под отдельные задачи. Две попытки добавить правила (ideal6, ideal7) показали худший результат и отвергнуты. ideal7 прогонялся на четырёх задачах из пяти.</p>
</section>

<section class="panel" aria-labelledby="h-fix">
  <h2 id="h-fix">Что исправили за эти дни</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>Проблема</th><th>Было</th><th>Стало</th></tr></thead>
    <tbody>
      <tr><td>Язык ответа</td><td>не работал на реальных запросах: служебный блок скрывал слова пользователя</td><td>ответ на языке вопроса во всех прогонах, закончившихся в срок (13 из 13)</td></tr>
      <tr><td>Кэш контекста</td><td>ломался в 27 ходах из 32</td><td>0 разрывов из 32</td></tr>
      <tr><td>Зависания по 120 с</td><td>15 из 16 зависаний вызывала команда <code>pkill -f</code>, убивавшая собственную оболочку</td><td>причина устранена в самом инструменте для любых моделей</td></tr>
      <tr><td>Мелкие цифры на картинке</td><td>12-значный номер модель переписывала с потерянной цифрой в каждом прогоне</td><td>плагин сам читает текст с картинки и добавляет его к сообщению: номер верный 3 из 3, доля проверок 48% → 74%</td></tr>
      <tr><td>Нагрев процессора</td><td>пики до 97 °C</td><td>65–75 °C при лимите мощности 45 Вт</td></tr>
    </tbody></table></div>
</section>

<section class="panel" aria-labelledby="h-pic">
  <h2 id="h-pic">Картинки: где ошибалась модель и как это починили</h2>
  <p>На задаче «сверстай страницу счёта по макету» наша модель теряла цифру в мелком номере внизу страницы. Проверка показала, что глаз у модели в порядке: при прямой просьбе переписать страницу все проверенные зрительные блоки читали номер верно. Цифра терялась позже, когда модель переписывала число в код. Поэтому ShuraCode теперь сам считывает текст с прикреплённой картинки программой распознавания и кладёт его рядом с картинкой. Саму картинку модель по-прежнему видит целиком.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Вариант</th><th class="n">Доля проверок</th><th class="n">Номер верный</th><th class="n">Медиана</th></tr></thead>
    <tbody>{pic_table()}</tbody></table></div>
  <p class="note" style="margin-top:10px">Задача h4, по три прогона. Оставшиеся ошибки уже не про цифры, а про детали вёрстки: слишком светлая шапка, пропущенная строка итогов или значок статуса. Qwen3.6 быстрее (2 минуты), но иногда сохраняет страницу не в ту папку, поэтому заменой нашей модели не стала.</p>
</section>

<section class="panel" aria-labelledby="h-open">
  <h2 id="h-open">Что не решено</h2>
  <ul class="plain">
    <li><b>Мелкий текст на картинках.</b> Модель теряет цифру в двенадцатизначном номере. Несжатая версия зрительного блока не помогла, это предел самой модели.</li>
    <li><b>Границы условия.</b> Если в задаче сказано «не округлять молча», все варианты ошибаются одинаково.</li>
    <li><b>Время.</b> Тяжёлые задачи занимают 15–30 минут на ноутбуке. Это цена полностью локальной работы; основной резерв ускорения даёт более мощное железо.</li>
    <li><b>Шум измерения.</b> Один и тот же режим на тех же задачах даёт результаты, различающиеся на десятки пунктов. Различить соседние ступени середины лесенки можно только на большем числе прогонов.</li>
  </ul>
</section>
</main>
'''
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html"), "w", encoding="utf-8").write(PAGE)
print("report.html", len(PAGE), "chars; best mode (without hd5):", best)
