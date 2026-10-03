import heapq
def plan(graph):
    deps={n:set(ds) for n,ds in graph.items()}
    for ds in list(deps.values()):
        for d in ds: deps.setdefault(d,set())
    left={n:set(ds) for n,ds in deps.items()}; reverse={n:[] for n in deps}
    for n,ds in deps.items():
        for d in ds: reverse[d].append(n)
    ready=[n for n,ds in left.items() if not ds]; heapq.heapify(ready); out=[]
    while ready:
        n=heapq.heappop(ready); out.append(n)
        for child in reverse[n]:
            left[child].discard(n)
            if not left[child]: heapq.heappush(ready,child)
    if len(out)!=len(left):
        remaining=set(left)-set(out); start=min(remaining); path=[]; seen={}
        def visit(n):
            if n in seen: return path[seen[n]:]+[n]
            seen[n]=len(path); path.append(n)
            for d in sorted(deps[n]):
                if d in remaining:
                    found=visit(d)
                    if found:return found
            path.pop();seen.pop(n);return None
        cycle=visit(start) or [start,start]
        err=ValueError("dependency cycle");err.cycle=cycle;raise err
    return out
