import json
import os
import re
import subprocess
import unittest

VENV_PY = "/home/hikkian/Desktop/AI/aiym/.venv/bin/python"

PW = r'''
import glob, json, os, sys
from playwright.sync_api import sync_playwright

url, shot = sys.argv[1], sys.argv[2]
errors = []

JS = """() => {
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const all = [...document.querySelectorAll('body *')].filter(e => e.getBoundingClientRect().width > 0);
  const deepest = (txt, exact) => {
    let best = null;
    for (const e of all) {
      const t = norm(e.innerText);
      if (exact ? t === txt : t.includes(txt)) { if (!best || best.contains(e)) best = e; }
    }
    return best;
  };
  const bgOf = e => {
    while (e && e !== document.documentElement) {
      const c = getComputedStyle(e).backgroundColor;
      if (c && !/rgba\\(.*,\\s*0\\)$/.test(c) && c !== 'transparent') return c;
      e = e.parentElement;
    }
    return 'rgb(255, 255, 255)';
  };
  const info = e => { if (!e) return null; const r = e.getBoundingClientRect(), cs = getComputedStyle(e);
    return {x: r.x, y: r.y, w: r.width, h: r.height, right: r.right, bottom: r.bottom, color: cs.color, bg: cs.backgroundColor,
            fw: parseInt(cs.fontWeight, 10), fs: parseFloat(cs.fontSize), br: cs.borderTopLeftRadius, pl: parseFloat(cs.paddingLeft),
            tag: e.tagName, text: norm(e.innerText)}; };
  const out = {};
  out.text = norm(document.body.innerText);
  out.overflow = document.documentElement.scrollWidth - window.innerWidth;
  out.pageH = document.documentElement.scrollHeight;
  // таблица
  const tables = [...document.querySelectorAll('table')];
  out.tables = tables.length;
  const t = tables.find(t => norm(t.innerText).includes('Монтаж')) || null;
  if (t) {
    const rows = [...t.querySelectorAll('tr')];
    out.rows = rows.map(r => {
      const cells = [...r.children].filter(c => /^(TD|TH)$/.test(c.tagName));
      const ri = info(r);
      return {text: norm(r.innerText), y: ri.y, bottom: ri.bottom, fw: ri.fw, cells: cells.map(c => ({t: norm(c.innerText), right: c.getBoundingClientRect().right, tr: (() => { const rg = document.createRange(); rg.selectNodeContents(c); return rg.getBoundingClientRect().right; })(),
              x: c.getBoundingClientRect().x, fw: parseInt(getComputedStyle(c).fontWeight, 10), fs: parseFloat(getComputedStyle(c).fontSize), bg: getComputedStyle(c).backgroundColor})),
              rowbg: bgOf(r), isTh: cells.length > 0 && cells.every(c => c.tagName === 'TH')};
    });
    out.table = info(t);
  }
  const names = ['Монтаж', 'Кабель', 'Настройка сети', 'Выезд', 'Итого к оплате'];
  out.ys = names.map(n => { const e = deepest(n, false); return e ? e.getBoundingClientRect().y : null; });
  out.badge = info(deepest('Оплачен частично', true));
  out.footerEl = info(deepest('БИН 123456789012', false));
  out.footerTag = (() => { const f = document.querySelector('footer'); return f ? info(f) : null; })();
  out.topBg = bgOf(document.elementFromPoint(500, 20));
  out.topGrad = (() => { let e = document.elementFromPoint(500, 20);
    while (e && e !== document.documentElement) {
      if (getComputedStyle(e).backgroundImage !== 'none') return true;
      const c = getComputedStyle(e).backgroundColor;
      if (c && !/rgba\\(.*,\\s*0\\)$/.test(c) && c !== 'transparent') return false;
      e = e.parentElement; }
    return false; })();
  out.h1 = info(deepest('Счёт № 117/2025', true));
  return out;
}"""

def launch(p):
    try:
        return p.chromium.launch()
    except Exception:
        ex = glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell'))
        ex += glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome'))
        return p.chromium.launch(executable_path=ex[0])

with sync_playwright() as p:
    b = launch(p)
    page = b.new_page(viewport={'width': 1000, 'height': 700})
    page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('requestfailed', lambda r: errors.append('requestfailed ' + r.url))
    page.goto(url)
    page.wait_for_timeout(300)
    res = page.evaluate(JS)
    try:
        from io import BytesIO
        from PIL import Image
        png = page.screenshot(clip={'x': 500, 'y': 20, 'width': 1, 'height': 1})
        res['topPx'] = list(Image.open(BytesIO(png)).convert('RGB').getpixel((0, 0)))
    except Exception as ex:
        res['topPx'] = None
    if shot:
        os.makedirs(os.path.dirname(shot), exist_ok=True)
        page.screenshot(path=shot, full_page=True)
    res['errors'] = errors
    b.close()
print('RESULT:' + json.dumps(res, ensure_ascii=False))
'''

STRINGS = [
    "Астра · Кабинет", "Алия К.", "Выйти", "Счёт № 117/2025", "от 12.03.2025", "Заказчик: ТОО «Астра Логистик»",
    "Оплачен частично", "Наименование", "Кол-во", "Цена", "Сумма",
    "Монтаж оборудования", "Кабель UTP, 305 м", "Настройка сети (срочно)", "Выезд специалиста",
    "45 000 ₸", "90 000 ₸", "38 500 ₸", "115 500 ₸", "120 000 ₸", "7 500 ₸", "30 000 ₸",
    "355 500 ₸", "Скидка 10%", "35 550 ₸", "Итого к оплате", "319 950 ₸",
    "© 2025 ТОО «Астра Логистик»", "БИН 123456789012", "Алматы, пр. Абая, 150/230",
]


def rgb(s):
    m = re.findall(r"[\d.]+", s)
    return tuple(float(x) for x in m[:3])


def alpha_zero(s):
    return s == "transparent" or bool(re.search(r",\s*0\)$", s))


def lum(c):
    r, g, b = c
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


class H(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = None
        cls.err = ""
        if not os.path.exists("index.html"):
            return
        shot_dir = os.environ.get("SHOT_DIR")
        shot = os.path.join(shot_dir, "h4.png") if shot_dir else ""
        r = subprocess.run([VENV_PY, "-c", PW, "file://" + os.path.abspath("index.html"), shot],
                           capture_output=True, text=True, timeout=90)
        cls.err = r.stderr[-500:]
        for line in r.stdout.splitlines():
            if line.startswith("RESULT:"):
                cls.res = json.loads(line[7:])

    def need(self):
        self.assertTrue(os.path.exists("index.html"), "нет index.html")
        self.assertIsNotNone(self.res, "playwright не отработал: " + self.err)
        return self.res

    def data_rows(self):
        r = self.need()
        self.assertIn("rows", r, "нет таблицы с позициями")
        return r["rows"]

    def test_files_no_external(self):
        self.assertTrue(os.path.exists("index.html"), "нет index.html")
        html = open("index.html", encoding="utf-8").read()
        self.assertIsNone(re.search(r'(src|href)\s*=\s*["\']https?://', html, re.I), "внешние ресурсы")
        self.assertIsNone(re.search(r"@import|url\(\s*[\"']?https?://", html, re.I))
        css = open("style.css", encoding="utf-8").read() if os.path.exists("style.css") else ""
        self.assertIsNone(re.search(r"url\(\s*[\"']?https?://|@import\s+url\(\s*[\"']?https?://", css, re.I))
        self.assertGreater(len(html) + len(css), 900)

    def test_no_console_errors_no_overflow(self):
        r = self.need()
        self.assertEqual(r["errors"], [])
        self.assertLessEqual(r["overflow"], 1, "горизонтальная прокрутка")

    def test_exact_strings(self):
        t = re.sub(r"\s+", " ", self.need()["text"])
        missing = [s for s in STRINGS if s not in t]
        self.assertFalse(missing, "нет строк: %s" % missing)
        self.assertRegex(t, r"[–−-] ?35 550 ₸", "скидка со знаком минус")

    def test_real_table_structure(self):
        rows = self.data_rows()
        hdr = [x for x in rows if "Наименование" in x["text"]]
        self.assertTrue(hdr, "нет строки заголовков")
        heads = [c["t"] for c in hdr[0]["cells"]]
        self.assertTrue(hdr[0]["isTh"], "заголовки не <th>")
        self.assertEqual(heads, ["№", "Наименование", "Кол-во", "Цена", "Сумма"])
        items = [x for x in rows if re.search(r"Монтаж оборудования|Кабель UTP|Настройка сети|Выезд специалиста", x["text"])]
        self.assertEqual(len(items), 4)
        self.assertEqual([x["cells"][0]["t"] for x in items], ["1", "2", "3", "4"])
        self.assertEqual([x["cells"][2]["t"] for x in items], ["2", "3", "1", "4"])
        ys = [x["y"] for x in items]
        self.assertEqual(ys, sorted(ys))
        for x in items:
            self.assertEqual(len(x["cells"]), 5, x["text"])

    def test_order_on_page(self):
        r = self.need()
        ys = r["ys"]
        self.assertNotIn(None, ys)
        self.assertEqual(ys, sorted(ys), ys)
        self.assertTrue(r["footerEl"], "нет подвала")
        self.assertGreater(r["footerEl"]["y"], r["table"]["bottom"])
        self.assertGreater(r["footerEl"]["y"], 500, "подвал не внизу окна")
        self.assertIsNotNone(r["h1"])
        self.assertLess(r["h1"]["y"], r["table"]["y"])

    def test_amount_columns_right_aligned(self):
        rows = self.data_rows()
        items = [x for x in rows if re.search(r"Монтаж оборудования|Кабель UTP|Настройка сети|Выезд специалиста", x["text"])]
        for col in (3, 4):
            rights = [x["cells"][col]["right"] for x in items]
            self.assertLessEqual(max(rights) - min(rights), 4, "колонка %d не выровнена по правому краю" % col)
        self.assertGreater(items[0]["cells"][4]["right"], items[0]["cells"][3]["right"] + 20)
        for x in items:
            for col in (3, 4):
                c = x["cells"][col]
                self.assertLessEqual(c["right"] - c["tr"], 20, "сумма не прижата вправо: " + c["t"])
        names_x = [x["cells"][1]["x"] for x in items]
        self.assertLessEqual(max(names_x) - min(names_x), 4)

    def test_highlighted_row(self):
        rows = self.data_rows()
        hl = [x for x in rows if "Настройка сети" in x["text"]][0]
        others = [x for x in rows if re.search(r"Монтаж оборудования|Кабель UTP|Выезд специалиста", x["text"])]

        def eff(x):
            c = x["cells"][1]["bg"]
            return c if not alpha_zero(c) else x["rowbg"]
        mine = rgb(eff(hl))
        self.assertTrue(all(rgb(eff(o)) != mine for o in others), "строка не отличается по фону")
        r_, g_, b_ = mine
        self.assertTrue(r_ >= 230 and g_ >= 200 and b_ <= g_ - 20, "ожидался светло-жёлтый фон, а не %s" % (mine,))
        self.assertGreaterEqual(hl["cells"][1]["fw"], 600, "выделенная строка не жирная")
        self.assertTrue(all(o["cells"][1]["fw"] < 600 for o in others), "остальные строки не должны быть жирными")

    def test_badge(self):
        b = self.need()["badge"]
        self.assertIsNotNone(b, "нет бейджа")
        self.assertFalse(alpha_zero(b["bg"]), "у бейджа нет фона")
        r_, g_, b_ = rgb(b["bg"])
        self.assertTrue(r_ > b_ + 20 and lum((r_, g_, b_)) > 0.8, "ожидался светло-оранжевый фон: %s" % b["bg"])
        cr, cg, cb = rgb(b["color"])
        self.assertTrue(cr > cb + 40 and lum((cr, cg, cb)) < 0.5, "ожидался тёмно-оранжевый текст: %s" % b["color"])
        br = b["br"]
        ok = br.endswith("%") or (br.endswith("px") and float(br[:-2]) >= min(12, b["h"] / 2 - 1))
        self.assertTrue(ok, "скругление: " + br)
        self.assertGreaterEqual(b["pl"], 6)
        self.assertGreater(b["x"], 600, "бейдж должен быть справа")
        self.assertLess(b["y"], self.need()["table"]["y"])

    def test_top_bar_and_total(self):
        r = self.need()
        if r.get("topGrad") and r.get("topPx"):  # градиент: смотрим реальный пиксель шапки
            self.assertLess(lum(tuple(r["topPx"])), 0.35, "шапка должна быть тёмной: %s" % r["topPx"])
        else:
            self.assertLess(lum(rgb(r["topBg"])), 0.35, "шапка должна быть тёмно-синей: %s" % r["topBg"])
        rows = self.data_rows()
        tot = [x for x in rows if "Итого к оплате" in x["text"]][0]
        item = [x for x in rows if "Монтаж оборудования" in x["text"]][0]
        last = tot["cells"][-1]
        self.assertGreater(last["fs"], item["cells"][-1]["fs"] + 1, "итог должен быть крупнее")
        self.assertGreaterEqual(last["fw"], 600)


if __name__ == "__main__":
    unittest.main()
