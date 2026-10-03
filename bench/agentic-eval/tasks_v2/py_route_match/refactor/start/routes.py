from urllib.parse import unquote
import re
def match_route(pattern, path):
    parts=pattern.strip('/').split('/') if pattern != '/' else []
    values=path.strip('/').split('/') if path != '/' else []
    names=[p[1:] for p in parts if p.startswith(':')]
    if any(not n for n in names) or len(set(names)) != len(names): raise ValueError("invalid parameter")
    if len(parts)!=len(values): return None
    out={}
    for want, got in zip(parts, values):
        try: decoded=unquote(got, errors="strict")
        except (UnicodeDecodeError, ValueError): return None
        if want.startswith(':'): out[want[1:]]=decoded
        elif want != got: return None
    return out
