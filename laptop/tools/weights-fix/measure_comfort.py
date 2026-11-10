import json,subprocess,time,urllib.request,sys,re
def sh(c): return subprocess.run(c,shell=True,capture_output=True,text=True).stdout.strip()
def avail_mb(): return int(sh("awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo"))
def rss_mb():
    out=sh("ps -C llama-server -o rss= | sort -n | tail -1"); return int(out)//1024 if out else 0
def vram(): return int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"))
def post(url,body,t=900):
    r=urllib.request.Request(url,json.dumps(body).encode(),{"Content-Type":"application/json"}); return json.load(urllib.request.urlopen(r,timeout=t))
def run(model):
    sh("systemctl --user restart ai-gateway"); time.sleep(15)
    idle=avail_mb()
    t=time.time(); post("http://127.0.0.1:8080/v1/chat/completions",{"model":model,"messages":[{"role":"user","content":"Скажи привет"}],"max_tokens":8,"stream":False,"chat_template_kwargs":{"enable_thinking":False}}); load=time.time()-t
    time.sleep(5); after_load=avail_mb(); rss0=rss_mb(); v0=vram()
    sp=[]
    for i in range(3):
        r=post("http://127.0.0.1:8090/v1/chat/completions",{"model":model,"messages":[{"role":"user","content":"Подробно объясни по-русски, как работает кэш процессора и почему он нужен. Не меньше шести абзацев."}],"max_tokens":450,"temperature":0,"stream":False,"chat_template_kwargs":{"enable_thinking":False}})
        tm=r.get("timings",{}); sp.append(round(tm.get("predicted_per_second",0),1))
    after=avail_mb(); rss1=rss_mb(); v1=vram()
    return dict(model=model,idle_avail_mb=idle,load_s=round(load),avail_after_load_mb=after_load,avail_after_gen_mb=after,rss_after_load_mb=rss0,rss_after_gen_mb=rss1,vram_mb=v1,tok_s=sp)
res=[run(m) for m in sys.argv[1:]]
for r in res: print(json.dumps(r,ensure_ascii=False))
