import json
import sqlite3
from note import main

def count_real(ledger, store):
    # Numbers only. Legacy rows with a missing author count as unknown.
    db = sqlite3.connect(ledger)
    try:
        cols = {r[1] for r in db.execute('PRAGMA table_info(events)')}
        authors = {a: 0 for a in ('human', 'local_model', 'closed_ai', 'unknown')}
        if 'author' in cols:
            authors.update({a: n for a, n in db.execute("SELECT COALESCE(author,'unknown'),COUNT(*) FROM events GROUP BY 1")})
        else:
            authors['unknown'] = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        clean = unclean = 0
        for (raw,) in db.execute("SELECT payload FROM events WHERE kind='outcome'"):
            if json.loads(raw).get('context_clean') is True: clean += 1
            else: unclean += 1
        print(*(authors[a] for a in ('human', 'local_model', 'closed_ai', 'unknown')), clean, unclean)
    finally:
        db.close()

import note
note.count_real = count_real
if __name__ == '__main__': main()
