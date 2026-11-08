import json, os, re, subprocess, unittest

VENV_PY = "/home/hikkian/Desktop/AI/aiym/.venv/bin/python"

PW = r'''
import glob, json, os, re, sys
from playwright.sync_api import sync_playwright

url, shot = sys.argv[1], sys.argv[2]
errors = []
res = {}

JS = """() => {
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const out = {};
  const all = [...document.querySelectorAll('body *')].filter(e => e.getBoundingClientRect().width > 0);
  function heading(name) {
    const re = new RegExp('^' + name + '(\\\\b|$)', 'i');
    let best = null;
    for (const e of all) {
      const t = norm(e.innerText);
      if (t.length <= name.length + 14 && re.test(t)) {
        if (!best || best.contains(e)) best = e;   // самый глубокий
      }
    }
    return best;
  }
  const ctaRe = /начать/i;
  const isCta = e => /^(BUTTON|A|INPUT)$/.test(e.tagName) || e.getAttribute('role') === 'button' || /btn|button|cta/i.test(e.className || '');
  const names = ['Старт', 'Про', 'Бизнес'];
  const cards = [];
  for (const n of names) {
    const h = heading(n);
    if (!h) { cards.push(null); continue; }
    let el = h, found = null;
    while (el && el !== document.body) {
      const t = norm(el.innerText);
      const others = names.filter(x => x !== n && new RegExp('(^|\\\\s)' + x + '(\\\\s|$)').test(t));
      const ctas = [...el.querySelectorAll('*')].filter(c => isCta(c) && ctaRe.test(c.innerText || c.value || ''));
      if (others.length) break;
      if (ctas.length) found = el;
      el = el.parentElement;
    }
    cards.push(found);
  }
  out.cards_found = cards.map(c => !!c);
  out.cards = cards.map(c => {
    if (!c) return null;
    const r = c.getBoundingClientRect(), cs = getComputedStyle(c);
    return {x: r.x, y: r.y, w: r.width, h: r.height, bg: cs.backgroundColor, bc: cs.borderTopColor, bw: cs.borderTopWidth,
            shadow: cs.boxShadow, tf: cs.transform, color: cs.color, outline: cs.outlineStyle,
            cta: [...c.querySelectorAll('*')].filter(e => isCta(e) && ctaRe.test(e.innerText || e.value || '')).length};
  });
  out.text = norm(document.body.innerText);
  out.title = document.title;
  out.cta_total = [...document.querySelectorAll('*')].filter(e => isCta(e) && ctaRe.test(e.innerText || e.value || '')).length;
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
    if shot:
        os.makedirs(os.path.dirname(shot), exist_ok=True)
        page.screenshot(path=shot, full_page=True)
    res['errors'] = errors
    b.close()
print('RESULT:' + json.dumps(res, ensure_ascii=False))
'''

FEATURES = ["5 ГБ хранилища", "1 проект", "Поддержка по почте", "100 ГБ хранилища", "10 проектов",
            "Приоритетная поддержка", "Автобэкапы", "1 ТБ хранилища", "Безлимит проектов", "Поддержка 24/7"]


class H(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = None
        cls.err = ""
        if not os.path.exists("index.html"):
            return
        shot_dir = os.environ.get("SHOT_DIR")
        shot = os.path.join(shot_dir, "t16.png") if shot_dir else ""
        r = subprocess.run([VENV_PY, "-c", PW, "file://" + os.path.abspath("index.html"), shot],
                           capture_output=True, text=True, timeout=120)
        cls.err = r.stderr[-500:]
        for line in r.stdout.splitlines():
            if line.startswith("RESULT:"):
                cls.res = json.loads(line[7:])

    def need(self):
        self.assertTrue(os.path.exists("index.html"), "нет index.html")
        self.assertIsNotNone(self.res, "playwright не отработал: " + self.err)
        return self.res

    def test_files(self):
        self.assertTrue(os.path.exists("index.html"), "нет index.html")
        self.assertTrue(os.path.exists("style.css"), "нет style.css")
        self.assertGreater(len(open("style.css", encoding="utf-8").read().strip()), 200)
        html = open("index.html", encoding="utf-8").read()
        self.assertRegex(html, r"style\.css")
        self.assertIsNone(re.search(r'(src|href)\s*=\s*["\']https?://', html, re.I), "внешние ресурсы")

    def test_no_console_errors(self):
        self.assertEqual(self.need()["errors"], [])

    def test_texts(self):
        t = self.need()["text"].lower()
        self.assertIn("shura cloud", t)
        flat = re.sub(r"\s+", "", t)
        self.assertIn("4900", flat)
        self.assertIn("14900", flat)
        for n in ("старт", "про", "бизнес"):
            self.assertIn(n, t)
        hits = sum(1 for f in FEATURES if re.sub(r"\s+", " ", f.lower()) in t)
        self.assertGreaterEqual(hits, 7, f"совпало пунктов списка: {hits}/10")

    def test_three_cards_in_row(self):
        r = self.need()
        self.assertEqual(r["cards_found"], [True, True, True], r["cards_found"])
        c = r["cards"]
        self.assertLess(c[0]["x"], c[1]["x"])
        self.assertLess(c[1]["x"], c[2]["x"])
        ys = [x["y"] for x in c]
        self.assertLess(max(ys) - min(ys), 90, "карточки не в один ряд")
        for x in c:
            self.assertGreater(x["w"], 150)
            self.assertLess(x["w"], 450)
            self.assertGreaterEqual(x["cta"], 1, "нет кнопки «Начать» в карточке")

    def test_middle_highlighted(self):
        c = self.need()["cards"]
        self.assertTrue(all(c))
        mid, a, b = c[1], c[0], c[2]
        def differs(k):
            return mid[k] != a[k] and mid[k] != b[k]
        score = [differs("bg"), differs("bc"), differs("bw"), differs("shadow"), differs("tf"), differs("color"),
                 mid["w"] > max(a["w"], b["w"]) + 5]  # высота зависит от числа пунктов, её не учитываем
        self.assertTrue(any(score), f"средняя карточка не выделена: {mid} vs {a} / {b}")

    def test_cta_button(self):
        self.assertGreaterEqual(self.need()["cta_total"], 3)


if __name__ == "__main__":
    unittest.main()
