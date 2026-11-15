"""Opt-in, local append-only experience ledger (LEDGER1).

The ledger deliberately stores task payloads separately from its event log so forget
can overwrite payload bytes while retaining an auditable tombstone.
"""
import datetime
import hashlib
import json
import os
import secrets
import sqlite3
from pathlib import Path
import lifelong_store

DEFAULT_PATH = Path.home() / ".local/share/shura/experience.sqlite"
PRIVACY = {"open", "personal", "work"}
OUTCOMES = {"accepted", "edited", "rejected"}


class LedgerError(ValueError):
    pass


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("SHURA_EXPERIENCE_DB") or DEFAULT_PATH).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self.db = sqlite3.connect(self.path)
        os.chmod(self.path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL,
            privacy TEXT NOT NULL, payload TEXT NOT NULL, forgotten_hash TEXT);
          CREATE TABLE IF NOT EXISTS consents(name TEXT PRIMARY KEY, enabled INTEGER NOT NULL, changed_at TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS blocks(block TEXT PRIMARY KEY, split TEXT NOT NULL, sealed INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(events)")}
        if "author" not in cols: self.db.execute("ALTER TABLE events ADD COLUMN author TEXT")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('salt', ?)", (secrets.token_hex(32),))
        self.db.commit()

    def close(self): self.db.close()

    def _append(self, kind, payload, privacy="work", require_consent=None, author="unknown"):
        author = lifelong_store.validate_author(author)
        if privacy not in PRIVACY: raise LedgerError("privacy must be open, personal or work")
        if require_consent and not self.consent(require_consent):
            raise LedgerError(f"consent required: {require_consent}")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode()) > 2_000_000: raise LedgerError("record exceeds 2 MB")
        cur = self.db.execute("INSERT INTO events(ts,kind,privacy,payload,author) VALUES (?,?,?,?,?)", (_now(), kind, privacy, encoded, author))
        self.db.commit()
        return {"id": cur.lastrowid, "kind": kind, "privacy": privacy, "author": author}

    def set_consent(self, name, enabled):
        if name not in {"capture", "growth", "weekly_acceptance", "exam"}: raise LedgerError("unknown consent")
        if type(enabled) is not bool: raise LedgerError("enabled must be boolean")
        self.db.execute("INSERT INTO consents VALUES (?,?,?) ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled, changed_at=excluded.changed_at", (name, int(enabled), _now()))
        self.db.commit()
        return {"name": name, "enabled": enabled}

    def consent(self, name):
        row = self.db.execute("SELECT enabled FROM consents WHERE name=?", (name,)).fetchone()
        return bool(row and row[0])

    def add_task(self, request, repo_snapshot=None, tests=None, repo=None, block=None, privacy="work", author="unknown"):
        if not isinstance(request, str) or not request.strip(): raise LedgerError("request is required")
        assigned = self.assign_split(block or repo or datetime.date.today().isoformat())
        result = self._append("task", {"request": request, "repo_snapshot": repo_snapshot, "tests": tests, "repo": repo,
                                         "block": block or repo or datetime.date.today().isoformat(), "split": assigned}, privacy, "capture", author)
        result["split"] = assigned
        return result

    def add_outcome(self, task_id, answer, status, test_results=None, privacy="work", context=None, author="unknown"):
        if status not in OUTCOMES: raise LedgerError("outcome must be accepted, edited or rejected")
        authors = []
        if isinstance(context, list):
            for item in context:
                if isinstance(item, str): authors.append(lifelong_store.validate_author(item))
                elif isinstance(item, dict): authors.append(lifelong_store.validate_author(item.get("author", "unknown")))
                else: authors.append("unknown")
        clean = bool(isinstance(context, list) and all(a in ("human", "local_model") for a in authors))
        return self._append("outcome", {"task_id": int(task_id), "answer": answer, "status": status,
                                         "test_results": test_results, "context_authors": sorted(set(authors)), "context_clean": clean}, privacy, "capture", author)

    def add_correction(self, task_id, answer, correction, privacy="work", author="unknown"):
        return self._append("correction_pair", {"task_id": int(task_id), "answer": answer, "correction": correction}, privacy, "capture", author)

    def add_gate_decision(self, decision, details=None, privacy="work"):
        return self._append("gate_decision", {"decision": decision, "details": details}, privacy)

    def set_growth_anchor(self, baseline, candidate, ci_low, ci_high, task_count, privacy="work"):
        if type(task_count) is not int or task_count < 1: raise LedgerError("task_count must be a positive integer")
        return self._append("growth_anchor", {"baseline": baseline, "candidate": candidate, "ci_low": ci_low,
                                               "ci_high": ci_high, "task_count": task_count}, privacy, "growth")

    def growth(self):
        anchors = [self.get(row[0]) for row in self.db.execute("SELECT id FROM events WHERE kind='growth_anchor' ORDER BY id DESC LIMIT 1")]
        anchor = anchors[0]["payload"] if anchors else None
        weekly = []
        if self.consent("weekly_acceptance"):
            rows = self.db.execute("SELECT substr(ts,1,10),payload FROM events WHERE kind='outcome' ORDER BY id").fetchall()
            by_week = {}
            for day, raw in rows:
                date = datetime.date.fromisoformat(day)
                week = (date - datetime.timedelta(days=date.weekday())).isoformat()
                value = json.loads(raw)
                total, accepted = by_week.get(week, (0, 0))
                by_week[week] = (total + 1, accepted + int(value["status"] == "accepted"))
            weekly = [{"week": week, "accepted_without_edits": accepted, "total": total,
                       "fraction": accepted / total if total else 0.0} for week, (total, accepted) in sorted(by_week.items())]
        return {"anchor": anchor, "weekly_acceptance": weekly, "weekly_acceptance_consent": self.consent("weekly_acceptance")}

    def add_exam_answer(self, exam_id, task_id, answer, privacy="work"):
        return self._append("exam_answer", {"exam_id": exam_id, "task_id": int(task_id), "answer": answer}, privacy, "exam")

    def add_pack_stack(self, states, privacy="work"):
        if not isinstance(states, list) or any(not isinstance(x, dict) or "verified" not in x for x in states):
            raise LedgerError("pack stack must be an ordered list of verified states")
        ranks = {"open": 0, "personal": 1, "work": 2}
        if privacy not in ranks: raise LedgerError("privacy must be open, personal or work")
        inherited = ranks[privacy]
        source_ids = {int(source_id) for state in states for source_id in state.get("source_event_ids", [])}
        for source_id in source_ids:
            source = self.db.execute("SELECT privacy FROM events WHERE id=? AND kind!='tombstone'", (source_id,)).fetchone()
            if source: inherited = max(inherited, ranks[source[0]])
        privacy = ("open", "personal", "work")[inherited]
        return self._append("pack_stack", {"states": states}, privacy)

    def assign_split(self, block):
        if not isinstance(block, str) or not block: raise LedgerError("block is required")
        row = self.db.execute("SELECT split FROM blocks WHERE block=?", (block,)).fetchone()
        if row: return row[0]
        salt = self.db.execute("SELECT value FROM meta WHERE key='salt'").fetchone()[0]
        digest = hashlib.sha256((salt + "\0" + block).encode()).digest()
        n = int.from_bytes(digest[:8], "big") % 10000
        # Two thirds train, one third exam; 10% of exam is sealed (about 3.3% overall).
        split = "train" if n < 6667 else ("sealed_exam" if n < 7000 else "exam")
        self.db.execute("INSERT INTO blocks(block,split,sealed) VALUES (?,?,?)", (block, split, int(split == "sealed_exam")))
        self.db.commit()
        return split

    def forget(self, event_id):
        row = self.db.execute("SELECT * FROM events WHERE id=?", (int(event_id),)).fetchone()
        if not row: raise LedgerError("record not found")
        if row["kind"] == "tombstone": raise LedgerError("record already forgotten")
        digest = hashlib.sha256(row["payload"].encode()).hexdigest()
        self.db.execute("BEGIN IMMEDIATE")
        # SQLite overwrite plus VACUUM removes the old cell from the database file when not in WAL mode.
        self.db.execute("UPDATE events SET payload='',kind='tombstone',forgotten_hash=? WHERE id=?", (digest, int(event_id)))
        self.db.commit()
        self.db.execute("VACUUM")
        return {"id": int(event_id), "forgotten": True, "sha256": digest}

    def get(self, event_id):
        row = self.db.execute("SELECT * FROM events WHERE id=?", (int(event_id),)).fetchone()
        if not row: raise LedgerError("record not found")
        if row["kind"] == "tombstone": return {"id": row["id"], "kind": "tombstone", "author": row["author"] or "unknown", "sha256": row["forgotten_hash"]}
        return {"id": row["id"], "ts": row["ts"], "kind": row["kind"], "privacy": row["privacy"], "author": row["author"] or "unknown", "payload": json.loads(row["payload"])}

    def list(self):
        return [self.get(r[0]) for r in self.db.execute("SELECT id FROM events ORDER BY id")]

    def export(self):
        # Privacy is inherited by derived records; only explicitly open records travel.
        rows = (self.get(r[0]) for r in self.db.execute("SELECT id FROM events ORDER BY id"))
        return "\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows
                         if x.get("kind") == "tombstone" or x.get("privacy") == "open") + "\n"

    def trainable(self, event_id):
        rec = self.get(event_id)
        if rec["kind"] == "tombstone": return False, "forgotten"
        if rec.get("author", "unknown") not in ("human", "local_model") and rec["kind"] in {"task", "correction_pair", "outcome", "exam_answer"}:
            return False, "author_closed_ai_or_unknown"
        if rec["kind"] == "outcome" and not rec["payload"].get("context_clean", False): return False, "context_not_clean"
        payload = rec.get("payload", {})
        block = payload.get("block")
        if rec["kind"] == "outcome":
            task = self.db.execute("SELECT payload FROM events WHERE id=? AND kind='task'", (payload.get("task_id"),)).fetchone()
            if task:
                block = json.loads(task[0]).get("block")
        if block:
            row = self.db.execute("SELECT split,sealed FROM blocks WHERE block=?", (block,)).fetchone()
            if row and (row["sealed"] or row["split"] in ("exam", "sealed_exam")): return False, "exam_or_sealed"
        return True, "trainable"

    def training_view(self):
        return (rec for rec in self.list() if self.trainable(rec["id"])[0])

    def pack_recipe(self, event_id):
        record = self.get(event_id)
        if record["kind"] != "pack_stack": raise LedgerError("not a pack stack")
        payload = record["payload"]
        erased = set()
        untrainable = set()
        referenced = set(payload.get("source_event_ids", []))
        referenced.update(source_id for state in payload["states"] for source_id in state.get("source_event_ids", []))
        for source_id in referenced:
            source = self.db.execute("SELECT kind FROM events WHERE id=?", (int(source_id),)).fetchone()
            if not source or source[0] == "tombstone": erased.add(int(source_id))
            elif not self.trainable(int(source_id))[0]: untrainable.add(int(source_id))
        states = payload["states"]
        if erased:
            states = [state for state in states if not ({int(x) for x in state.get("source_event_ids", [])} & erased)]
        if untrainable:
            states = [state for state in states if not ({int(x) for x in state.get("source_event_ids", [])} & untrainable)]
        return {"states": states, "reproducible": True, "omitted_erased_source_ids": sorted(erased), "omitted_untrainable_source_ids": sorted(untrainable)}

    def stats(self):
        n = self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        authors = {a: self.db.execute("SELECT COUNT(*) FROM events WHERE COALESCE(author,'unknown')=?", (a,)).fetchone()[0] for a in lifelong_store.AUTHOR_CLASSES}
        outcomes = self.db.execute("SELECT payload FROM events WHERE kind='outcome'").fetchall()
        return {"records": n, "authors": authors, "clean_outcomes": sum(bool(json.loads(r[0]).get("context_clean")) for r in outcomes), "unclean_outcomes": sum(not bool(json.loads(r[0]).get("context_clean")) for r in outcomes), "database_bytes": self.path.stat().st_size, "bytes_per_day_estimate": 100 * 1024,
                "ssd_write_bytes_per_day_estimate": 100 * 1024, "consents": {r[0]: bool(r[1]) for r in self.db.execute("SELECT name,enabled FROM consents")}}

    def migrate_capture(self, legacy_db):
        """One-way import of LL6 capture signals; duplicate source ids are ignored."""
        if not self.consent("capture"): raise LedgerError("consent required: capture")
        src = sqlite3.connect(f"file:{Path(legacy_db).resolve()}?mode=ro", uri=True)
        try:
            rows = src.execute("SELECT j.id,m.title,m.body,m.scope FROM journal j JOIN memory m ON m.id=j.memory_id WHERE j.event='signal_captured' ORDER BY j.id").fetchall()
        finally: src.close()
        imported = 0
        for jid, title, body, scope in rows:
            marker = f"ll6-capture:{jid}"
            if self.db.execute("SELECT 1 FROM events WHERE json_extract(payload,'$.migration_id')=?", (marker,)).fetchone(): continue
            self._append("task", {"request": title, "captured_text": body, "legacy_scope": scope, "migration_id": marker}, "personal")
            imported += 1
        return {"imported": imported, "seen": len(rows)}
