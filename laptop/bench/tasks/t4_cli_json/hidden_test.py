import json, subprocess, sys, unittest
def run(*a):
    return subprocess.run([sys.executable, "todo.py", *a], capture_output=True, text=True, check=True).stdout
class H(unittest.TestCase):
    def test_json(self):
        d = json.loads(run("list", "--json"))
        self.assertEqual(d, [{"id": 1, "title": "Купить молоко", "done": False},
                             {"id": 2, "title": "Позвонить маме", "done": True},
                             {"id": 3, "title": "Write report", "done": False}])
    def test_plain(self):
        self.assertEqual(run("list"), "[ ] 1. Купить молоко\n[x] 2. Позвонить маме\n[ ] 3. Write report\n")
