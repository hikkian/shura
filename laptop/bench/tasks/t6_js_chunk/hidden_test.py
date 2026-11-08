import subprocess, unittest, pathlib
JS = """
import assert from "node:assert/strict";
import { chunk } from "./lib/chunk.js";
assert.deepEqual(chunk([1,2,3,4], 2), [[1,2],[3,4]]);
assert.deepEqual(chunk([1,2,3], 5), [[1,2,3]]);
assert.deepEqual(chunk([], 3), []);
assert.throws(() => chunk([1], 0), RangeError);
"""
class H(unittest.TestCase):
    def test_js(self):
        pathlib.Path("_hidden.mjs").write_text(JS)
        r = subprocess.run(["node", "_hidden.mjs"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
