FAMILY = {
    "name": "f21_ledger", "lang": "python",
    "implement": {
        "prompt": """Write the module `ledger.py` (standard library only; amounts are integers of the smallest unit, no floats) with the class `Ledger` and the exceptions `LedgerError(Exception)` and `InsufficientFunds(LedgerError)`.

A double-entry ledger. Accounts have a kind: "asset", "expense" (normal balance on the debit side) or "liability", "income", "equity" (normal
balance on the credit side).
  open_account(name, kind)       create an account (kind must be one of the five, otherwise ValueError; a duplicate name raises LedgerError).
  post(entries, memo="", key=None) -> int   entries is a list of (account, side, amount) with side "debit" or "credit" and amount an int > 0
                                 (ValueError otherwise). The total debits must equal the total credits (otherwise LedgerError) and all
                                 accounts must exist (otherwise LedgerError); at least two entries are required. The whole transaction is
                                 applied or nothing is. Returns the transaction id (1, 2, 3, ...). If `key` is given and a transaction with
                                 that key was already posted, nothing is posted and the ORIGINAL id is returned (idempotency).
  balance(name) -> int           the balance in the account's normal direction (an asset's balance = debits - credits; a liability's =
                                 credits - debits). Unknown account: LedgerError.
  transfer(src, dst, amount, memo="", key=None) -> int   move money between two ASSET accounts (debit dst, credit src); if that would make
                                 src's balance negative raise InsufficientFunds and change nothing.
  reverse(txn_id, memo="") -> int  post a new transaction with every entry's side flipped; the original stays in the history. Reversing a
                                 transaction twice, or a reversal, raises LedgerError; an unknown id raises LedgerError.
  trial_balance() -> dict        {"debits": total of all debit amounts posted, "credits": total of all credit amounts, "balanced": bool}.
  history(name=None) -> list     transactions as dicts {"id", "memo", "entries"} (entries as tuples), oldest first; with a name only those
                                 touching that account.
""",
        "start": {"ledger.py": '"""A double-entry ledger (to be written)."""\n\n\nclass LedgerError(Exception):\n    pass\n\n\nclass InsufficientFunds(LedgerError):\n    pass\n\n\nclass Ledger:\n    def __init__(self):\n        raise NotImplementedError\n'},
        "solution": {"ledger.py": '''"""A double-entry ledger."""

KINDS = {"asset": "debit", "expense": "debit", "liability": "credit", "income": "credit", "equity": "credit"}


class LedgerError(Exception):
    pass


class InsufficientFunds(LedgerError):
    pass


class Ledger:
    def __init__(self):
        self._kind, self._bal = {}, {}      # balance stored as debits - credits
        self._txns, self._keys, self._reversed, self._reversals = [], {}, set(), set()

    def open_account(self, name, kind):
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        if name in self._kind:
            raise LedgerError(f"account {name!r} exists")
        self._kind[name], self._bal[name] = kind, 0

    def _check(self, entries):
        if len(entries) < 2:
            raise LedgerError("at least two entries")
        debit = credit = 0
        for acct, side, amount in entries:
            if side not in ("debit", "credit") or isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
                raise ValueError("bad entry")
            if acct not in self._kind:
                raise LedgerError(f"unknown account {acct!r}")
            if side == "debit":
                debit += amount
            else:
                credit += amount
        if debit != credit:
            raise LedgerError("unbalanced transaction")

    def post(self, entries, memo="", key=None):
        entries = [tuple(e) for e in entries]
        if key is not None and key in self._keys:
            return self._keys[key]
        self._check(entries)
        for acct, side, amount in entries:
            self._bal[acct] += amount if side == "debit" else -amount
        txn_id = len(self._txns) + 1
        self._txns.append({"id": txn_id, "memo": memo, "entries": entries})
        if key is not None:
            self._keys[key] = txn_id
        return txn_id

    def balance(self, name):
        if name not in self._kind:
            raise LedgerError(f"unknown account {name!r}")
        raw = self._bal[name]
        return raw if KINDS[self._kind[name]] == "debit" else -raw

    def transfer(self, src, dst, amount, memo="", key=None):
        if key is not None and key in self._keys:
            return self._keys[key]
        for n in (src, dst):
            if n not in self._kind:
                raise LedgerError(f"unknown account {n!r}")
            if self._kind[n] != "asset":
                raise LedgerError("transfers are between asset accounts")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("amount must be a positive int")
        if self.balance(src) < amount:
            raise InsufficientFunds(f"{src!r} has {self.balance(src)}, needs {amount}")
        return self.post([(dst, "debit", amount), (src, "credit", amount)], memo, key)

    def reverse(self, txn_id, memo=""):
        if not isinstance(txn_id, int) or txn_id < 1 or txn_id > len(self._txns):
            raise LedgerError("unknown transaction")
        if txn_id in self._reversed or txn_id in self._reversals:
            raise LedgerError("already reversed, or is a reversal")
        flipped = [(a, "credit" if s == "debit" else "debit", amt) for a, s, amt in self._txns[txn_id - 1]["entries"]]
        new_id = self.post(flipped, memo or f"reversal of {txn_id}")
        self._reversed.add(txn_id)
        self._reversals.add(new_id)
        return new_id

    def trial_balance(self):
        d = sum(a for t in self._txns for _, s, a in t["entries"] if s == "debit")
        c = sum(a for t in self._txns for _, s, a in t["entries"] if s == "credit")
        return {"debits": d, "credits": c, "balanced": d == c}

    def history(self, name=None):
        if name is not None and name not in self._kind:
            raise LedgerError(f"unknown account {name!r}")
        return [dict(t) for t in self._txns if name is None or any(e[0] == name for e in t["entries"])]
'''},
    },
    "hidden": {"test_ledger.py": '''import unittest
from ledger import InsufficientFunds, Ledger, LedgerError


def make():
    l = Ledger()
    for n, k in [("cash", "asset"), ("bank", "asset"), ("sales", "income"), ("rent", "expense"), ("loan", "liability"), ("capital", "equity")]:
        l.open_account(n, k)
    return l


class Accounts(unittest.TestCase):
    def test_open(self):
        l = make()
        with self.assertRaises(LedgerError):
            l.open_account("cash", "asset")
        with self.assertRaises(ValueError):
            l.open_account("x", "bogus")
        with self.assertRaises(LedgerError):
            l.balance("nope")

    def test_normal_balances(self):
        l = make()
        l.post([("cash", "debit", 100), ("sales", "credit", 100)])
        l.post([("rent", "debit", 30), ("cash", "credit", 30)])
        l.post([("cash", "debit", 50), ("loan", "credit", 50)])
        self.assertEqual((l.balance("cash"), l.balance("sales"), l.balance("rent"), l.balance("loan")), (120, 100, 30, 50))


class Posting(unittest.TestCase):
    def test_ids_and_atomicity(self):
        l = make()
        self.assertEqual(l.post([("cash", "debit", 10), ("sales", "credit", 10)]), 1)
        for bad, exc in [([("cash", "debit", 10), ("sales", "credit", 9)], LedgerError), ([("cash", "debit", 10)], LedgerError),
                         ([("cash", "debit", 10), ("ghost", "credit", 10)], LedgerError), ([("cash", "debit", 0), ("sales", "credit", 0)], ValueError),
                         ([("cash", "debit", 1.5), ("sales", "credit", 1.5)], ValueError), ([("cash", "left", 5), ("sales", "credit", 5)], ValueError),
                         ([("cash", "debit", 10), ("sales", "credit", 5), ("bank", "credit", 5), ("ghost", "debit", 0)], ValueError)]:
            with self.assertRaises(exc):
                l.post(bad)
        self.assertEqual(l.balance("cash"), 10)
        self.assertEqual(l.post([("cash", "debit", 1), ("sales", "credit", 1)]), 2)

    def test_idempotency_key(self):
        l = make()
        a = l.post([("cash", "debit", 10), ("sales", "credit", 10)], key="k1")
        b = l.post([("cash", "debit", 99), ("sales", "credit", 99)], key="k1")
        self.assertEqual(a, b)
        self.assertEqual(l.balance("cash"), 10)
        self.assertEqual(len(l.history()), 1)

    def test_multi_entry(self):
        l = make()
        l.post([("cash", "debit", 60), ("bank", "debit", 40), ("sales", "credit", 100)])
        self.assertEqual((l.balance("cash"), l.balance("bank"), l.balance("sales")), (60, 40, 100))


class Transfers(unittest.TestCase):
    def test_transfer_and_insufficient(self):
        l = make()
        l.post([("cash", "debit", 100), ("capital", "credit", 100)])
        l.transfer("cash", "bank", 70, "to bank")
        self.assertEqual((l.balance("cash"), l.balance("bank")), (30, 70))
        with self.assertRaises(InsufficientFunds) as cm:
            l.transfer("cash", "bank", 31)
        self.assertIsInstance(cm.exception, LedgerError)
        self.assertEqual((l.balance("cash"), l.balance("bank")), (30, 70))
        self.assertEqual(len(l.history()), 2)

    def test_transfer_rules(self):
        l = make()
        l.post([("cash", "debit", 100), ("capital", "credit", 100)])
        with self.assertRaises(LedgerError):
            l.transfer("cash", "sales", 5)
        with self.assertRaises(LedgerError):
            l.transfer("cash", "ghost", 5)
        for bad in (0, -5, 1.5, True):
            with self.assertRaises(ValueError):
                l.transfer("cash", "bank", bad)

    def test_transfer_idempotent(self):
        l = make()
        l.post([("cash", "debit", 100), ("capital", "credit", 100)])
        a = l.transfer("cash", "bank", 60, key="t")
        self.assertEqual(l.transfer("cash", "bank", 60, key="t"), a)
        self.assertEqual(l.balance("bank"), 60)


class Reversal(unittest.TestCase):
    def test_reverse(self):
        l = make()
        t = l.post([("cash", "debit", 100), ("sales", "credit", 100)])
        r = l.reverse(t)
        self.assertEqual((l.balance("cash"), l.balance("sales")), (0, 0))
        self.assertEqual(len(l.history()), 2)
        self.assertEqual(l.history()[1]["id"], r)
        for again in (t, r):
            with self.assertRaises(LedgerError):
                l.reverse(again)
        for unknown in (0, 99, "1"):
            with self.assertRaises(LedgerError):
                l.reverse(unknown)

    def test_trial_balance_and_history(self):
        l = make()
        t = l.post([("cash", "debit", 100), ("sales", "credit", 100)], memo="sale")
        l.post([("rent", "debit", 20), ("bank", "credit", 20)])
        l.reverse(t)
        tb = l.trial_balance()
        self.assertEqual(tb, {"debits": 220, "credits": 220, "balanced": True})
        self.assertEqual([h["id"] for h in l.history("cash")], [1, 3])
        self.assertEqual(l.history("cash")[0]["memo"], "sale")
        self.assertEqual(l.history("cash")[0]["entries"], [("cash", "debit", 100), ("sales", "credit", 100)])
        with self.assertRaises(LedgerError):
            l.history("ghost")
'''},
    "bugfix": {
        "symptom": "A ledger (ledger.py) reports the wrong balance for liability accounts: after posting a credit of 50 to the account \"loan\" its balance should be 50 (normal side is credit) but it shows -50.",
        "bug": [("ledger.py", '        return raw if KINDS[self._kind[name]] == "debit" else -raw', '        return raw')],
        "repro": {"test_repro.py": 'import unittest\nfrom ledger import Ledger\n\n\nclass Repro(unittest.TestCase):\n    def test_liability_balance_is_positive(self):\n        l = Ledger()\n        l.open_account("cash", "asset")\n        l.open_account("loan", "liability")\n        l.post([("cash", "debit", 50), ("loan", "credit", 50)])\n        self.assertEqual(l.balance("loan"), 50)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
