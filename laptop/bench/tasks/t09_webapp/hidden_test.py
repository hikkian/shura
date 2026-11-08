import json, os, subprocess, sys, unittest

VENV_PY = "/home/hikkian/Desktop/AI/aiym/.venv/bin/python"

PW = r'''
import glob, json, os, re, sys
from playwright.sync_api import sync_playwright

url, shot = sys.argv[1], sys.argv[2]
res = {}
errors = []

FIND_ROW = """([text, tag]) => {
  const boxes = [...document.querySelectorAll('input[type=checkbox]')];
  for (const cb of boxes) {
    let el = cb.parentElement;
    while (el && el !== document.body) {
      const n = el.querySelectorAll('input[type=checkbox]').length;
      if (n > 1) break;
      if (el.textContent.includes(text) || [...el.querySelectorAll('input')].some(i => i.value === text)) {
        el.setAttribute('data-eval-row', tag); cb.setAttribute('data-eval-cb', tag); return true;
      }
      el = el.parentElement;
    }
  }
  return false;
}"""

WIDE = """(tag) => {
  const cb = document.querySelector('[data-eval-cb=' + tag + ']');
  if (!cb) return false;
  let el = cb.parentElement;
  while (el.parentElement && el.parentElement !== document.body && el.parentElement.querySelectorAll('input[type=checkbox]').length <= 1) el = el.parentElement;
  el.setAttribute('data-eval-wide', tag); return true;
}"""

def launch(p):
    try:
        return p.chromium.launch()
    except Exception:
        ex = glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell'))
        ex += glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome'))
        return p.chromium.launch(executable_path=ex[0])

def left(page):
    t = page.inner_text('body')
    m = re.search(r'(?:осталось|остал[а-я]*|left|remaining)\D{0,12}?(\d+)', t, re.I)
    if not m:
        m = re.search(r'(\d+)\s*(?:дел|задач|осталось|left|remaining|items? left)', t, re.I)
    return int(m.group(1)) if m else None

def toggle(page, text):
    # клик по чекбоксу (приложение могло перерисовать DOM), затем перечитываем состояние
    if not page.evaluate(FIND_ROW, [text, 'tg']):
        return None
    page.locator('[data-eval-cb=tg]').first.click()
    page.wait_for_timeout(150)
    if not page.evaluate(FIND_ROW, [text, 'tg2']):
        return None
    return page.locator('[data-eval-cb=tg2]').first.is_checked()

def add(page, text, how):
    inp = None
    for sel in ['input[type=text]', 'input:not([type])', 'input[type=search]', 'textarea']:
        loc = page.locator(sel)
        for i in range(loc.count()):
            if loc.nth(i).is_visible():
                inp = loc.nth(i); break
        if inp: break
    if inp is None:
        raise RuntimeError('no text input')
    inp.fill(text)
    if how == 'enter':
        inp.press('Enter')
    else:
        btn = None
        cands = page.locator('button, input[type=submit], input[type=button], [role=button]')
        for i in range(cands.count()):
            c = cands.nth(i)
            if not c.is_visible(): continue
            label = (c.inner_text() if c.evaluate('e=>e.tagName') != 'INPUT' else (c.get_attribute('value') or ''))
            if re.search(r'добав|add|\+|создат|ok|ок', label, re.I):
                btn = c; break
        if btn is None:
            btn = cands.first
        btn.click()
    page.wait_for_timeout(100)

with sync_playwright() as p:
    b = launch(p)
    ctx = b.new_context(viewport={'width': 900, 'height': 700})
    page = ctx.new_page()
    page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.set_default_timeout(5000)
    page.goto(url)
    page.wait_for_timeout(200)
    try:
        add(page, 'Купить хлеб', 'button')
        add(page, 'Позвонить маме', 'enter')
        add(page, 'Сдать отчёт', 'button')
        body = page.inner_text('body')
        res['added'] = all(s in body for s in ['Купить хлеб', 'Позвонить маме', 'Сдать отчёт'])
        res['checkboxes'] = page.locator('input[type=checkbox]').count()
        res['left_3'] = left(page)
        st = toggle(page, 'Позвонить маме')
        res['row_found'] = st is not None
        res['checked'] = st is True
        res['left_after_check'] = left(page)
        if shot:
            os.makedirs(os.path.dirname(shot), exist_ok=True)
            page.screenshot(path=shot, full_page=True)
        # reload keeps items and state
        page.reload()
        page.wait_for_timeout(200)
        body = page.inner_text('body')
        res['persist_text'] = all(s in body for s in ['Купить хлеб', 'Позвонить маме', 'Сдать отчёт'])
        res['persist_left'] = left(page)
        ok = page.evaluate(FIND_ROW, ['Позвонить маме', 'r2'])
        res['persist_checked'] = bool(ok) and page.locator('[data-eval-cb=r2]').first.is_checked()
        # снимаем отметку
        if ok:
            toggle(page, 'Позвонить маме')
            res['left_after_uncheck'] = left(page)
        # delete
        ok = page.evaluate(FIND_ROW, ['Купить хлеб', 'r1'])
        res['row1_found'] = ok
        if ok:
            page.evaluate(WIDE, 'r1')
            row = page.locator('[data-eval-wide=r1]').first
            btns = row.locator('button, [role=button]')
            target = None
            for i in range(btns.count()):
                t = btns.nth(i).inner_text() + ' ' + (btns.nth(i).get_attribute('aria-label') or '') + ' ' + (btns.nth(i).get_attribute('title') or '')
                if re.search(r'[×✕✖xх✗🗑❌]|удал|delete|del|remove|trash', t, re.I):
                    target = btns.nth(i); break
            if target is None and btns.count():
                target = btns.nth(btns.count() - 1)
            if target is not None:
                target.click()
                page.wait_for_timeout(100)
        body = page.inner_text('body')
        res['deleted'] = 'Купить хлеб' not in body and 'Позвонить маме' in body
        res['left_after_delete'] = left(page)
        page.reload()
        page.wait_for_timeout(200)
        body = page.inner_text('body')
        res['deleted_persists'] = 'Купить хлеб' not in body and 'Сдать отчёт' in body
    except Exception as e:
        res['exception'] = repr(e)
    res['console_errors'] = errors
    b.close()
print('RESULT:' + json.dumps(res, ensure_ascii=False))
'''


class H(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = None
        path = os.path.abspath("index.html")
        if not os.path.exists(path):
            return
        shot_dir = os.environ.get("SHOT_DIR")
        shot = os.path.join(shot_dir, "t09.png") if shot_dir else ""
        r = subprocess.run([VENV_PY, "-c", PW, "file://" + path, shot], capture_output=True, text=True, timeout=120)
        for line in r.stdout.splitlines():
            if line.startswith("RESULT:"):
                cls.res = json.loads(line[7:])
        cls.err = r.stderr[-500:]

    def need(self):
        self.assertTrue(os.path.exists("index.html"), "нет index.html")
        self.assertIsNotNone(self.res, "playwright не отработал: " + getattr(self, "err", ""))
        self.assertNotIn("exception", self.res, self.res.get("exception"))
        return self.res

    def test_no_external_deps(self):
        self.assertTrue(os.path.exists("index.html"), "нет index.html")
        src = open("index.html", encoding="utf-8").read()
        import re
        self.assertIsNone(re.search(r'(src|href)\s*=\s*["\']https?://', src, re.I), "внешние зависимости")
        self.assertIn("localStorage", src)

    def test_add_and_counter(self):
        r = self.need()
        self.assertTrue(r["added"], r)
        self.assertEqual(r["checkboxes"], 3, r)
        self.assertEqual(r["left_3"], 3, r)

    def test_toggle(self):
        r = self.need()
        self.assertTrue(r["row_found"], r)
        self.assertTrue(r["checked"], r)
        self.assertEqual(r["left_after_check"], 2, r)
        self.assertEqual(r["left_after_uncheck"], 3, r)

    def test_persist(self):
        r = self.need()
        self.assertTrue(r["persist_text"], r)
        self.assertEqual(r["persist_left"], 2, r)
        self.assertTrue(r["persist_checked"], r)

    def test_delete(self):
        r = self.need()
        self.assertTrue(r["row1_found"], r)
        self.assertTrue(r["deleted"], r)
        self.assertTrue(r["deleted_persists"], r)
        self.assertEqual(r["left_after_delete"], 2, r)

    def test_no_console_errors(self):
        r = self.need()
        self.assertEqual(r["console_errors"], [], r)


if __name__ == "__main__":
    unittest.main()
