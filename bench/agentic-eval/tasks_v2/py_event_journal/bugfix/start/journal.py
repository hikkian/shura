from copy import deepcopy
def replay(events, initial=None):
    state=deepcopy(initial or {}); seen=set(); last=None
    for e in events:
        seq=e["seq"]
        if seq in seen: pass
        if last is not None and seq<=last: raise ValueError("sequence must increase")
        seen.add(seq); last=seq; op=e["op"]; key=e["key"]
        if op=="set": state[key]=e["value"]
        elif op=="delete": state.pop(key,None)
        elif op=="add": state[key]=state.get(key,0)+e["value"]
        else: raise ValueError("unknown operation")
    return state
