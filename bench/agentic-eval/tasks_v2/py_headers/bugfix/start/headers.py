def parse_headers(lines):
    out={}; last=None
    for raw in lines:
        line=raw.rstrip("\r\n")
        if not line.strip(): continue
        if line[0] in " \t":
            if last is None: raise ValueError("orphan continuation")
            val=out[last]; val[-1]=line.strip(); continue
        if ":" not in line: raise ValueError("missing colon")
        name,value=line.split(":",1); name=name.strip().lower(); value=value.strip()
        if not name: raise ValueError("empty name")
        if name in out: out[name].append(value)
        else: out[name]=[value]
        last=name
    return {k:(v[0] if len(v)==1 else v) for k,v in out.items()}
