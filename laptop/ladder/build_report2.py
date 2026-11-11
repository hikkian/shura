#!/usr/bin/env python3
"""build_report2.py -> report.html: the management page as a project story (what we built, what we achieved), ladder as one block.
Numbers: ladder / Luna comparison / picture tests come from final_data.json + results (via the helpers of build_report.py); the project facts are
typed here from project_facts.md (each has its source there). Run: python3 -B final_tables.py && REPORT_FINAL=1 python3 -B build_report2.py"""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "build_report.py"), encoding="utf8").read()
ns = {"__file__": os.path.join(HERE, "build_report.py")}
exec(compile(src[:src.index("PAGE = f'''")], "build_report_head", "exec"), ns)
pc, sec, e = ns["pc"], ns["sec"], ns["e"]
ladder_rows, versus_rows, pic_table = ns["ladder_rows"], ns["versus_rows"], ns["pic_table"]
LAD, T3, luna_hd, cx, o5, lm, lf, spread = ns["LAD"], ns["T3"], ns["luna_hd"], ns["cx"], ns["o5"], ns["lm"], ns["lf"], ns["spread"]
css = src[src.index("<style>") + 7:src.index("</style>")].replace("{{", "{").replace("}}", "}")
EXTRA = """
.hero { padding:6px 0 2px } .hero h1 { font-size:clamp(30px,6vw,46px) }
.kpis4 { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin-top:22px }
.kpi4 { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px }
.kpi4 b { display:block; font:500 30px/1.1 var(--mono); color:var(--ours) } .kpi4 span { color:var(--ink2); font-size:13.5px; display:block; margin-top:4px }
.stack { display:grid; gap:8px; margin:14px 0 6px }
.layer { display:grid; grid-template-columns:150px 1fr; gap:6px 14px; padding:10px 12px; border-left:3px solid var(--ours); background:color-mix(in srgb,var(--panel) 70%,var(--bg)) }
.layer b { font:500 14px var(--mono) } .layer span { color:var(--ink2); font-size:14px }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px; margin-top:10px }
.card { border:1px solid var(--line); border-radius:6px; padding:12px 14px; background:var(--panel) } .card h3 { margin:0 0 4px; font-size:15px }
.card .ba { font:500 13px var(--mono); color:var(--ours) } .card p { margin:4px 0 0; font-size:13.5px; color:var(--ink2) }
.tl { list-style:none; margin:10px 0 0; padding:0; display:grid; gap:0 } .tl li { display:grid; grid-template-columns:78px 1fr; gap:14px; padding:8px 0; border-top:1px solid var(--line) } .tl li:first-child { border-top:0 }
.tl time { font:500 13px var(--mono); color:var(--ink2) }
.mrow { display:grid; grid-template-columns:150px 1fr 52px; gap:10px; align-items:center; padding:5px 0 } .mrow .n { font-size:14px } .mrow .v { font:500 13px var(--mono); text-align:right }
.mt { position:relative; height:18px; background:color-mix(in srgb,var(--line) 55%,transparent); border-radius:3px } .mt b { position:absolute; inset:0 auto 0 0; background:var(--ink2); border-radius:3px 0 0 3px }
.mrow.me .mt b { background:var(--ours) } .mrow.codex .mt b { background:var(--luna) }
@media (max-width:520px) { .layer { grid-template-columns:1fr } .mrow { grid-template-columns:112px 1fr 46px } }
"""
def mrow(name, val, cls=""):
    return f'<div class="mrow {cls}"><div class="n">{e(name)}</div><div class="mt"><b style="width:{100*val/57:.1f}%"></b></div><div class="v">{val}/57</div></div>'
FINAL = os.environ.get("REPORT_FINAL") == "1"
BANNER = "" if FINAL else '<div class="panel" style="border-left:4px solid var(--warn)"><b class="warn">Промежуточная версия.</b></div>'
PAGE = f"""<title>Итоги проекта ShuraCode</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}{EXTRA}</style>
<main>
{BANNER}
<header class="hero">
  <h1>Локальный ИИ-программист, который работает на ноутбуке без облака</h1>
  <p class="lead">ShuraCode читает код, правит файлы, запускает тесты и отвечает по-русски. Всё идёт на собственном компьютере: код и данные никуда не уходят, интернет для работы не нужен.</p>
  <div class="kpis4">
    <div class="kpi4"><b>55 из 57</b><span>заданий выполняет наш агент, столько же, сколько платный Codex (19 заданий, по 3 прогона)</span></div>
    <div class="kpi4"><b>200 000</b><span>токенов контекста помещается на ноутбуке с видеокартой всего на 6 ГБ</span></div>
    <div class="kpi4"><b>44–46</b><span>токена в секунду на этом ноутбуке (быстрее на 40% благодаря спекулятивному декодированию)</span></div>
    <div class="kpi4"><b>0 из 32</b><span>сбоев кэша контекста между ходами; раньше ломался в 27 ходах из 32</span></div>
  </div>
</header>

<section class="panel" aria-labelledby="h-built">
  <h2 id="h-built">Что мы построили</h2>
  <p>Полный стек, от окна в терминале до собственной сборки движка. Каждый слой написан или доработан нами под одну цель: чтобы большая модель с 35 млрд параметров работала быстро и аккуратно на обычной видеокарте.</p>
  <div class="stack">
    <div class="layer"><b>ShuraCode</b><span>агент в терминале на базе OpenCode: понимает задачу, читает проект, вызывает инструменты, помнит сессии и ваши указания</span></div>
    <div class="layer"><b>Шлюз</b><span>сам загружает модель по запросу и выгружает при простое, переключает текст и картинки, следит за памятью, защищает ответ: язык и опечатки</span></div>
    <div class="layer"><b>Движок shura.cpp</b><span>наша сборка llama.cpp: сжатый кэш контекста (TurboQuant), ускорение генерации, умная выгрузка частей модели на процессор</span></div>
    <div class="layer"><b>Модель</b><span>Ornith-1.5 35B-A3B: из 35 млрд параметров на каждый токен работают около 3 млрд, поэтому она работает на 6 ГБ видеопамяти вместе с оперативной памятью ноутбука</span></div>
  </div>
  <div class="cards">
    <div class="card"><h3>Пять режимов усилия</h3><p>От быстрого ответа без раздумий до максимума: чем выше режим, тем дольше модель думает и тем тщательнее проверяет себя.</p></div>
    <div class="card"><h3>Подсказки и шаблон чата</h3><p>Свой шаблон и системная инструкция, которые держат ответ на языке пользователя и стабильный кэш. Инструкция сжата с 13,9 до 9,2 тысяч токенов.</p></div>
    <div class="card"><h3>Зрение и чтение текста с картинок</h3><p>Агент понимает скриншоты и макеты. Текст с картинки дополнительно читается программой распознавания, чтобы цифры и надписи не терялись.</p></div>
    <div class="card"><h3>Защита ответа</h3><p>Если пользователь пишет по-русски, а ответ ушёл в английский, шлюз его исправит. Опечатки в русском тексте ловятся по словарю.</p></div>
    <div class="card"><h3>Безопасные команды</h3><p>Исправлена ловушка, из-за которой команда остановки процесса убивала собственную оболочку и подвешивала агента на две минуты.</p></div>
    <div class="card"><h3>Установщик и память</h3><p>Установка с проверкой диска и контрольных сумм; постоянная память между сессиями и восстановление сессии за секунды.</p></div>
  </div>
</section>

<section class="panel" aria-labelledby="h-choice">
  <h2 id="h-choice">Как выбирали модель: по задачам, а не по рекламе</h2>
  <p>Мы сравнили несколько моделей на 19 настоящих агентных заданиях: правка кода, длинные документы, макеты, поиск в интернете, честность. Каждое задание по три раза, скрытые тесты агент не видит.</p>
  {mrow("Codex (облако)", 55, "codex")}
  {mrow("Ornith (наш выбор)", 55, "me")}
  {mrow("Qwen3.6, шаблон Sharp", 53)}
  {mrow("Tiel-Coder", 51)}
  {mrow("Qwen3.6, стандартный шаблон", 50)}
  <p class="note" style="margin-top:10px">Столбики Codex, Qwen и Tiel сняты 7 ноября. Результат Ornith 55 из 57 получен 11 ноября повторным полным прогоном на текущей рабочей настройке (режим medium, без единого таймаута, медиана 78 с против 49 с у Codex). Этот набор мы использовали и при настройке, поэтому цифра не вполне независима: независимые проверки ниже. Единственное слабое место: задание про README (1 из 3). Ornith выбрана за силу на трудных заданиях и честность: когда причину проблемы установить нельзя, она так и говорит, а не выдумывает. Qwen3.6 оставлена вторым вариантом для русских текстов и скриншотов.</p>
</section>

<section class="panel" aria-labelledby="h-honest">
  <h2 id="h-honest">Честная проверка на задачах, которых модель не видела</h2>
  <p>Чтобы не обманывать себя, мы написали 11 новых заданий, а скрытые тесты зафиксировали до первого запуска. Для тяжёлых задач есть эталонное решение, «наивное» и «среднее»: так видно, что задача действительно различает уровни. Независимая проверка нашла пробелы в описаниях трёх задач, мы закрыли их до запуска.</p>
  <h3 style="font-size:16px;margin:14px 0 4px">Режимы на тяжёлых задачах</h3>
  <div class="legend"><span><i class="sw-dot"></i>результат отдельного повтора</span><span><i class="sw-tick"></i>среднее со всеми шестью задачами</span></div>
  {ladder_rows()}
  <p class="note" style="margin-top:12px">Столбец: средняя доля пройденных проверок без самой тяжёлой задачи, которая на любом режиме упирается в лимит времени. Режим high теперь равен прежнему xhigh, старый high убран: он не дал выигрыша над medium. Это решение принято после просмотра результатов, поэтому его строка оставлена. Результат одного режима колеблется между повторами до {100*spread:.0f} пунктов, поэтому крупные разрывы надёжны (fast против остальных, max против medium), а соседние ступени различить нельзя, хотя средние идут вверх.</p>
  <h3 style="font-size:16px;margin:18px 0 4px">Рядом с Luna (Codex) на новых задачах</h3>
  <div class="legend"><span><i class="sw-luna"></i>Luna</span><span><i class="sw-ours"></i>ShuraCode</span></div>
  {versus_rows()}
  <p style="margin-top:12px">В среднем у Luna {pc(cx['mean_frac'])} проверок, у нас {pc(o5['mean_frac'])}; на тяжёлых задачах у Luna {pc(luna_hd.get('mean_frac'))}, у нашего лучшего режима {pc(lm['mean_frac'])}. Скорость сравнивать нужно с оговоркой: Luna работает в облаке, мы полностью локально на ноутбуке, поэтому медиана задачи {sec(cx['median_secs'])} против {sec(o5['median_secs'])}.</p>
</section>

<section class="panel" aria-labelledby="h-fixed">
  <h2 id="h-fixed">Проблемы, которые мы нашли и решили</h2>
  <p>Часть улучшений вскрылась только при честной проверке. Вот главные.</p>
  <div class="cards">
    <div class="card"><h3>Язык ответа</h3><div class="ba">не работал → работает</div><p>Служебный английский блок в сообщении скрывал слова пользователя. Теперь русский вопрос получает русский ответ во всех завершённых прогонах.</p></div>
    <div class="card"><h3>Кэш контекста</h3><div class="ba">27 сбоев из 32 → 0</div><p>Подсказка менялась на каждом ходу и заставляла модель перечитывать весь диалог. Теперь она неизменна, ответы быстрее.</p></div>
    <div class="card"><h3>Зависания агента</h3><div class="ba">16 раз по 2 минуты → 0</div><p>Одна команда останавливала саму себя. Причину нашли по логам и убрали для всех моделей.</p></div>
    <div class="card"><h3>Цифры с картинки</h3><div class="ba">0 из 3 → 3 из 3</div><p>Модель теряла цифру в 12-значном номере при переписывании. Теперь текст с картинки читается отдельно и кладётся рядом с ней.</p></div>
    <div class="card"><h3>Знак ₸ и макеты</h3><div class="ba">не читался → читается</div><p>Подобрали разрешение картинки для модели; без этого знак тенге пропадал.</p></div>
    <div class="card"><h3>Потерянное время на тяжёлых задачах</h3><div class="ba">0% → 100% к 15-й минуте</div><p>Модель долго думала и ничего не сохраняла. Добавили общее правило «сначала сохрани первую рабочую версию, потом улучшай», и на тестах готовый результат появляется в разы раньше.</p></div>
    <div class="card"><h3>Граничные условия задач</h3><div class="ba">56% → 83% проверок</div><p>Модель перестала пропускать значения, которые нужно отклонять (например, молча округляемые). Правило «выпиши все названные границы и проверь каждую» работает.</p></div>
    <div class="card"><h3>Опечатки по-русски</h3><div class="ba">запас буквы 1,73 → 2,38</div><p>Нашли, что главная причина в сжатии последнего слоя модели, и точечно подняли его точность без обучения: +0,18 ГБ и ни мегабайта свободной памяти потеряно, скорость та же. Плюс шлюз исправляет оставшиеся опечатки по словарю.</p></div>
    <div class="card"><h3>Перегрев ноутбука</h3><div class="ba">97 °C → 65–75 °C</div><p>Ограничили мощность процессора на время долгих прогонов.</p></div>
    <div class="card"><h3>Подгонка под тесты</h3><div class="ba">убрана</div><p>Правила в подсказках, написанные под конкретные задачи, заменены общими принципами, новые задачи написаны заранее.</p></div>
  </div>
  <p class="note" style="margin-top:12px">Мы также проверили и отвергли несколько своих идей (две версии дополнительных правил, более точный зрительный блок, ограничение длины размышлений): они не дали улучшения.</p>
</section>

<section class="panel" aria-labelledby="h-pic">
  <h2 id="h-pic">Картинки: где ошибалась модель</h2>
  <p>На задаче «сверстай страницу счёта по макету» модель теряла цифру в мелком номере внизу. Проверка показала, что зрение у неё в порядке: при прямой просьбе переписать страницу номер читался верно во всех проверенных вариантах. Цифра терялась позже, при копировании в код, поэтому текст с картинки теперь читается отдельно.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Вариант</th><th class="n">Доля проверок</th><th class="n">Номер верный</th><th class="n">Медиана</th></tr></thead>
    <tbody>{pic_table()}</tbody></table></div>
  <p class="note" style="margin-top:10px">Задача h4, по три прогона. Остаются ошибки вёрстки: слишком светлая шапка, пропущенный значок статуса.</p>
</section>

<section class="panel" aria-labelledby="h-time">
  <h2 id="h-time">Как это развивалось</h2>
  <ul class="tl">
    <li><time>27.09</time><span>первый выпуск: локальный запуск модели 35B на видеокарте, команда <code>shura</code>, установщик</span></li>
    <li><time>28.10</time><span>первый выбор моделей по 112 агентным прогонам</span></li>
    <li><time>05.11</time><span>переписана системная подсказка, заработали режимы усилия, картинки без перезагрузки модели</span></li>
    <li><time>07.11</time><span>сравнение с Codex на 19 заданиях, выбрана модель Ornith</span></li>
    <li><time>09.11</time><span>честные проверки: новые задачи с тестами, написанными заранее; кэш контекста починен</span></li>
    <li><time>11.11</time><span>повторный полный прогон 19 заданий на рабочей настройке: 55 из 57, как у Codex; точечно улучшена точность последнего слоя модели; новые правила подсказки</span></li>
    <li><time>10.11</time><span>чтение текста с картинок, защита от опечаток, пять режимов усилия в рабочей настройке</span></li>
  </ul>
</section>

<section class="panel" aria-labelledby="h-next">
  <h2 id="h-next">Что дальше</h2>
  <ul class="plain">
    <li><b>Опечатки по-русски у корня.</b> Причина в трёхбитном сжатии модели. Первый шаг: взять более точную версию той же модели и замерить опечатки. План и источники подготовлены.</li>
    <li><b>Расширения агента.</b> Исследуем навыки, плагины и подключаемые инструменты, которые облегчат работу и поднимут качество, не раздувая контекст.</li>
    <li><b>Больше повторов.</b> Чтобы уверенно различать соседние режимы, нужно 5–6 повторов вместо двух.</li>
    <li><b>Больше повторов тяжёлых задач.</b> Новые правила подсказки проверены на небольшом числе прогонов (по 2–4), эффект большой, но его стоит закрепить сериями из 5–6 повторов.</li>
    <li><b>Скорость.</b> Тяжёлые задачи теперь укладываются во времени, но Luna в облаке всё равно в разы быстрее; цель следующего этапа сократить время до первой рабочей версии ещё сильнее.</li>
  </ul>
  <p class="note" style="margin-top:12px">Ограничения, о которых надо знать: тяжёлые задачи занимают 15–30 минут на ноутбуке; на условиях с границами («не округлять молча») ошибаются все модели, и Codex тоже.</p>
</section>
</main>
"""
open(os.path.join(HERE, "report.html"), "w", encoding="utf8").write(PAGE)
print("report.html", len(PAGE), "chars")
