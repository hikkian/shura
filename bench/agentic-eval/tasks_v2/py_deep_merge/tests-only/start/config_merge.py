from copy import deepcopy
def deep_merge(base, overlay):
    if not isinstance(base,dict) or not isinstance(overlay,dict): raise TypeError("roots must be dicts")
    out=deepcopy(base)
    for k,v in overlay.items():
        if k in out and isinstance(out[k],dict) and isinstance(v,dict): out[k]=deep_merge(out[k],v)
        else: out[k]=deepcopy(v)
    return out
