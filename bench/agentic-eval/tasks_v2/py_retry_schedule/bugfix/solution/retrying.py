import math
def delay_for(attempt, base, cap, jitter=0, sample=0.5):
    if isinstance(attempt,bool) or not isinstance(attempt,int) or attempt<1: raise ValueError("attempt")
    vals=(base,cap,jitter,sample)
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in vals) or base<0 or cap<base or not 0<=jitter<=1 or not 0<=sample<=1: raise ValueError("range")
    raw=min(cap, base*2**(attempt-1))
    return raw*(1-jitter+2*jitter*sample)
