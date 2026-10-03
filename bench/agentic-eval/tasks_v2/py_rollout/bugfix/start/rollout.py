import hashlib
def enabled(flag,user_id,percentage,salt=""):
    if not isinstance(flag,str) or not flag or not isinstance(user_id,str) or not user_id: raise ValueError("flag and user id required")
    if isinstance(percentage,bool) or not isinstance(percentage,(int,float)) or not 0<=percentage<=100: raise ValueError("percentage")
    raw=f"{salt}\0{flag}\0{user_id}".encode()
    bucket=int.from_bytes(hashlib.sha256(raw).digest()[:4],"big")%10000
    return bucket < percentage*100
