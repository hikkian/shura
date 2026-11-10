import difflib
import random
import unittest

from patchlib import apply_patch, PatchError, ConflictError


def diff(a, b, n=3):
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "a/f", "b/f", n=n))


class H(unittest.TestCase):
    def test_simple_and_untouched_rest(self):
        old = "one\ntwo\nthree\nfour\nfive\n"
        p = "--- a/f\n+++ b/f\n@@ -2,3 +2,3 @@\n two\n-three\n+THREE\n four\n"
        self.assertEqual(apply_patch(old, p), "one\ntwo\nTHREE\nfour\nfive\n")

    def test_offset_with_lines_added_above(self):
        base = "".join("line%d\n" % i for i in range(1, 21))
        new = base.replace("line10\n", "TEN\n")
        p = diff(base, new)
        shifted = "intro1\nintro2\nintro3\n" + base
        self.assertEqual(apply_patch(shifted, p), "intro1\nintro2\nintro3\n" + new)
        trimmed = "".join(base.splitlines(True)[4:])
        self.assertEqual(apply_patch(trimmed, p), "".join(new.splitlines(True)[4:]))

    def test_drift_from_previous_hunk_decides_between_candidates(self):
        orig = ["f%d" % i for i in range(40)]
        orig[6:9] = ["c1", "A", "c2"]          # ханк 1: в заголовке стоит на 5 строк раньше (индекс 1)
        orig[17:20] = ["x", "y", "z"]          # «ловушка»: ближе к номеру из заголовка ханка 2
        orig[24:27] = ["x", "y", "z"]          # настоящее место: заголовок (индекс 19) + смещение 5
        patch = ("@@ -2,3 +2,3 @@\n c1\n-A\n+B\n c2\n"
                 "@@ -20,3 +20,3 @@\n x\n-y\n+Y\n z\n")
        want = list(orig)
        want[7] = "B"
        want[25] = "Y"
        self.assertEqual(apply_patch("".join(l + "\n" for l in orig), patch), "".join(l + "\n" for l in want))

    def test_multi_hunk_numbering_and_growth(self):
        old = "".join("l%d\n" % i for i in range(1, 31))
        new_l = old.splitlines(True)
        new_l[2:3] = ["a1\n", "a2\n", "a3\n"]        # +2 строки
        del new_l[20:22]                             # -2 строки
        new_l.append("tail\n")
        new = "".join(new_l)
        p = diff(old, new)
        self.assertGreaterEqual(p.count("@@ -"), 2)
        self.assertEqual(apply_patch(old, p), new)
        self.assertEqual(apply_patch(new, p, reverse=True), old)

    def test_generated_patches_roundtrip_and_shift(self):
        rnd = random.Random(7)
        alphabet = ["a\n", "b\n", "c\n", "\n", "  x\n", "-- s\n", "++ t\n", "@@ z\n", "d \n"]
        for case in range(150):
            a = [rnd.choice(alphabet) if rnd.random() < 0.7 else "u%d\n" % rnd.randrange(50) for _ in range(rnd.randrange(0, 30))]
            b = list(a)
            for _ in range(rnd.randrange(1, 5)):
                op = rnd.choice("ird")
                k = rnd.randrange(0, len(b) + 1)
                if op == "i":
                    b[k:k] = [rnd.choice(alphabet) for _ in range(rnd.randrange(1, 4))]
                elif op == "d" and b:
                    del b[k:k + rnd.randrange(1, 4)]
                elif b:
                    b[min(k, len(b) - 1)] = "r%d\n" % rnd.randrange(50)
            A, B = "".join(a), "".join(b)
            if A == B:
                continue
            for n in (0, 1, 3):
                p = diff(A, B, n)
                self.assertEqual(apply_patch(A, p), B, (case, n, p))
                self.assertEqual(apply_patch(B, p, reverse=True), A, (case, n, p))
                if n == 3 and ",0 +" not in p:   # со сдвигом — только ханки с непустой старой стороной
                    pad = "".join("pad%d\n" % i for i in range(rnd.randrange(1, 6)))
                    self.assertEqual(apply_patch(pad + A, p), pad + B, (case, n, "shift"))

    def test_conflicts(self):
        old = "".join("l%d\n" % i for i in range(1, 21))
        p = ("@@ -2,3 +2,3 @@\n l2\n-l3\n+L3\n l4\n"
             "@@ -15,3 +15,3 @@\n l15\n-l16\n+L16\n l17\n")
        self.assertEqual(apply_patch(old, p).count("L"), 2)
        for bad, hunk in [(old.replace("l16\n", "l16 \n"), 2), (old.replace("l4\n", "l4x\n"), 1),
                          (old.replace("l15\n", ""), 2)]:
            with self.assertRaises(ConflictError) as cm:
                apply_patch(bad, p)
            self.assertEqual(cm.exception.hunk, hunk)
        once = apply_patch(old, p)
        with self.assertRaises(ConflictError):
            apply_patch(once, p)
        # ханки в обратном порядке по тексту не должны применяться, перекрывая друг друга
        p2 = "@@ -5,2 +5,2 @@\n l5\n-l6\n+X6\n@@ -3,2 +3,2 @@\n l3\n-l4\n+X4\n"
        with self.assertRaises(ConflictError) as cm:
            apply_patch(old, p2)
        self.assertEqual(cm.exception.hunk, 2)

    def test_pure_insertions_and_creation(self):
        old = "a\nb\nc\n"
        self.assertEqual(apply_patch(old, "@@ -2,0 +3,2 @@\n+x\n+y\n"), "a\nb\nx\ny\nc\n")
        self.assertEqual(apply_patch(old, "@@ -0,0 +1 @@\n+first\n"), "first\na\nb\nc\n")
        self.assertEqual(apply_patch(old, "@@ -3,0 +4 @@\n+last\n"), "a\nb\nc\nlast\n")
        self.assertEqual(apply_patch("", "@@ -0,0 +1,2 @@\n+a\n+b\n"), "a\nb\n")
        self.assertEqual(apply_patch("a\nb\n", "@@ -1,2 +0,0 @@\n-a\n-b\n"), "")
        self.assertEqual(apply_patch("a\nb\nc\nd\n", "@@ -2,2 +1,0 @@\n-b\n-c\n"), "a\nd\n")
        self.assertEqual(apply_patch("a\nd\n", "@@ -1,0 +2,2 @@\n+b\n+c\n"), "a\nb\nc\nd\n")
        # нулевой счётчик + смещение: вставка идёт после той же строки
        self.assertEqual(apply_patch("zz\na\nb\n", "@@ -1,0 +2 @@\n+x\n"), "zz\nx\na\nb\n")
        self.assertEqual(apply_patch("a\nb\n", "@@ -1,0 +2 @@\n+x\n"), "a\nx\nb\n")
        self.assertEqual(apply_patch("a\nx\nb\n", "@@ -1,0 +2 @@\n+x\n", reverse=True), "a\nb\n")

    def test_no_newline_at_end_of_file(self):
        p = "@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+b\n"
        self.assertEqual(apply_patch("a\nb", p), "a\nb\n")
        self.assertEqual(apply_patch("a\nb\n", p, reverse=True), "a\nb")
        with self.assertRaises(ConflictError):
            apply_patch("a\nb\n", p)
        p2 = "@@ -1,2 +1,2 @@\n a\n-b\n+c\n\\ No newline at end of file\n"
        self.assertEqual(apply_patch("a\nb\n", p2), "a\nc")
        p3 = "@@ -1,2 +1,2 @@\n a\n b\n\\ No newline at end of file\n"
        self.assertEqual(apply_patch("a\nb", p3), "a\nb")
        with self.assertRaises(ConflictError):
            apply_patch("a\nb\n", p3)
        p4 = "@@ -1,2 +1,3 @@\n a\n-b\n\\ No newline at end of file\n+b\n+c\n"
        self.assertEqual(apply_patch("a\nb", p4), "a\nb\nc\n")

    def test_header_variants(self):
        self.assertEqual(apply_patch("a\nb\n", "@@ -2 +2 @@\n-b\n+B\n"), "a\nB\n")
        self.assertEqual(apply_patch("a\nb\n", "@@ -1 +1,2 @@ def f():\n-a\n+a\n+a2\n"), "a\na2\nb\n")
        junk = "diff --git a/f b/f\nindex 1..2 100644\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+A\n\n\n"
        self.assertEqual(apply_patch("a\nb\n", junk), "A\nb\n")
        self.assertEqual(apply_patch("a\nb\n", "@@ -1 +1 @@\n-a\n+A"), "A\nb\n")   # без \n в конце патча

    def test_body_lines_that_look_like_headers(self):
        old = "keep\n-- comment\n++ plus\n@@ -1 +1 @@\nend\n"
        p = ("--- a/f\n+++ b/f\n@@ -1,5 +1,5 @@\n keep\n--- comment\n+++ plus2\n"
             "-++ plus\n+-- comment\n @@ -1 +1 @@\n end\n")
        self.assertEqual(apply_patch(old, p), "keep\n++ plus2\n-- comment\n@@ -1 +1 @@\nend\n")

    def test_blank_context_lines_and_odd_characters(self):
        old = "a\n\nb\n\nc\n"
        p = "@@ -1,5 +1,5 @@\n a\n\n b\n\n-c\n+C\n"          # пустая строка вместо ' '
        self.assertEqual(apply_patch(old, p), "a\n\nb\n\nC\n")
        crlf = "a\r\nb\r\nc\r\n"
        pc = "@@ -1,3 +1,3 @@\n a\r\n-b\r\n+B\r\n c\r\n"
        self.assertEqual(apply_patch(crlf, pc), "a\r\nB\r\nc\r\n")
        odd = "x\x0cy\nmid dle\nz\n"
        po = "@@ -1,3 +1,3 @@\n x\x0cy\n-mid dle\n+MID DLE\n z\n"
        self.assertEqual(apply_patch(odd, po), "x\x0cy\nMID DLE\nz\n")
        ws = "a \nb\n"
        with self.assertRaises(ConflictError):
            apply_patch(ws, "@@ -1,2 +1,2 @@\n a\n-b\n+B\n")

    def test_ties_prefer_earlier_and_nearest(self):
        old = "k\nk\nk\nk\nk\n"
        # хотим заменить одну строку k; ожидаемая позиция — 3-я (индекс 2), совпадает точно
        self.assertEqual(apply_patch(old, "@@ -3 +3 @@\n-k\n+K\n"), "k\nk\nK\nk\nk\n")
        # ожидаемая позиция за концом -> ближайшая (последняя)
        self.assertEqual(apply_patch("k\nk\n", "@@ -9 +9 @@\n-k\n+K\n"), "k\nK\n")
        # равное расстояние: 2 строки "m" вокруг ожидаемого индекса -> берётся более ранняя
        self.assertEqual(apply_patch("m\nq\nm\n", "@@ -2 +2 @@\n-m\n+M\n"), "M\nq\nm\n")

    def test_malformed_patches(self):
        bad = ["", "no hunks here\n", "@@ nonsense @@\n x\n", "@@ -1,2 +1,2 @@\n a\n",              # ханк оборван
               "@@ -1 +1 @@\n a\n-b\n+c\n",                                                      # лишние строки
               "@@ -1 +1 @@\n-a\n+A\nstray text\n",                                              # мусор после ханка
               "@@ -1,2 +1,1 @@\n-a\n-b\n+c\n+d\n", "@@ -1 +1 @@\n*a\n",
               "@@ -1,2 +1,2 @@\n a\n\\ No newline at end of file\n b\n"]                       # маркер в середине
        for p in bad:
            with self.assertRaises(PatchError, msg=repr(p)) as cm:
                apply_patch("a\nb\n", p)
            self.assertNotIsInstance(cm.exception, ConflictError, repr(p))
