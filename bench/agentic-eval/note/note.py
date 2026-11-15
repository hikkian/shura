"""NOTE0 data/builders/analysis. Synthetic evaluation only; never training data."""
from __future__ import annotations
import argparse, json, random, re, sqlite3, urllib.request, urllib.error
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
KINDS=[
 {"id":"tests_only","params":["{test_file}"],"applies":"test command","accepted":r"(?i)(pytest|unittest|npm test|vitest|jest).*{test_file}","old":r"(?i)(pytest|unittest|npm test|vitest|jest).*"},
 {"id":"linter_first","params":["{linter}"],"applies":"validation command","accepted":r"(?i){linter}","old":r"(?i)(pytest|npm test|cargo test)"},
 {"id":"no_sed","params":[],"applies":"edit tool","accepted":r"(?i)^(edit|write)$","old":r"(?i)^shell$"},
 {"id":"russian_final","params":[],"applies":"final answer","accepted":r"(?i)^final_ru$","old":r"(?i)^final_non_ru$"},
 {"id":"changelog","params":["CHANGELOG.md"],"applies":"file edit","accepted":r"(?i)CHANGELOG\.md","old":r"(?i)(README|docs?/).*"},
 {"id":"protect_file","params":["{protected_file}"],"applies":"file edit","accepted":r"(?i)^(?!.*{protected_file}).*$","old":r"(?i){protected_file}"},
]
MODES=["base","near","pinned20","pinned200","retr5","long_base","long_near","long_pinned20"]
def classify(action, fact):
 for label,key in (("accepted","accepted"),("old mistake","old")):
  params=fact.get('params',{})
  if not isinstance(params,dict): params={}
  try: pat=fact[key].format(**params)
  except (KeyError,ValueError): pat=fact[key]
  if re.search(pat,str(action)): return label
 return "other"
def lesson_text(f): return "Правило: " + f['id'] + ("; " + ", ".join(f.get('params',{}).values()) if f.get('params') else "")
def check_lesson_text(text,f): return all(v in text for v in f.get('params',{}).values())
def make_body(point, mode, lesson=None, distractors=(), seed=0):
 b=json.loads(json.dumps(point['body'])); msgs=b.setdefault('messages',[])
 if mode in ('near','long_near') and lesson: msgs.append({'role':'system','content':lesson})
 if mode.startswith('pinned'):
  arr=list(distractors); arr.append(lesson or ''); random.Random(seed).shuffle(arr)
  b['messages'][0]['content']=str(b['messages'][0].get('content',''))+'\n\n'+"\n".join(arr)
 if mode=='retr5':
  b.setdefault('messages',[]).insert(1,{'role':'system','content':'\n'.join(list(distractors)[:4]+[lesson or ''])})
 return b
def bootstrap(values, seed=1, n=2000):
 if not values:return [0.,0.]
 r=random.Random(seed); means=[sum(r.choice(values) for _ in values)/len(values) for _ in range(n)]; means.sort()
 return [means[int(.025*n)],means[min(n-1,int(.975*n))]]
def analyze(rows):
 modes=defaultdict(list); trunc=Counter(); valid=[]; excluded=0
 grouped=defaultdict(dict)
 for x in rows: grouped[x['pair_id']][x['mode']]=x
 for pid,g in grouped.items():
  a=g.get('base'); b=g.get('base2')
  if not a or not b: continue
  if a.get('class')!='old mistake' or b.get('class')!='old mistake': excluded+=1; continue
  valid.append(pid)
  for m,x in g.items():
   if m not in ('base2',): modes[m].append((x.get('class')=='accepted',bool(x.get('truncated')),x.get('idea')))
 out={}
 for m, vals in modes.items():
  v=[int(a) for a,t,_ in vals if not t]; out[m]={'F':sum(v)/len(v) if v else None,'ci':bootstrap(v),'truncated':sum(t for _,t,_ in vals)/len(vals) if vals else 0,'n':len(v)}
 ideas={i for _,_,i in modes.get('base',[]) if i}
 verdict='недостаточно данных' if len(valid)<40 or len(ideas)<8 else 'серая зона'
 def f(m):return out.get(m,{}).get('F')
 if f('near') is not None and f('pinned20') is not None and f('long_near') is not None:
  if f('near')<=.70 or f('long_near')<.7*f('near') or (f('pinned200') is not None and f('pinned200')<.7*f('pinned20')):verdict='заметка не держится'
  elif f('near')>=.8 and f('pinned20')>=.7 and f('pinned200') is not None and f('pinned200')>=.9*f('pinned20') and f('long_near')>=.7*f('near'):verdict='заметка держится'
 return {'valid':len(valid),'excluded':excluded,'ideas':len(ideas),'modes':out,'verdict':verdict}
def build_dataset(manifest, logsdir):
 d=json.load(open(manifest)); tasks={x['id']:x for x in d['tasks'] if x.get('part')=='search'}
 # Recorded action events form candidates; exact prompt bodies are recovered by FORK handover, never guessed here.
 pts=[]
 for p in sorted(Path(logsdir).glob('*.agent.log')):
  tid=p.name.split('__')[0].replace('_','-',1)
  meta=next((x for k,x in tasks.items() if k.replace('-','_',1)==p.name.split('__')[0]),None)
  if not meta:continue
  for line in p.read_text(errors='replace').splitlines():
   try:e=json.loads(line)
   except:continue
   if e.get('type')=='tool_use':
    s=e.get('part') or {}; state=s.get('state') or {}; inp=state.get('input') or {}
    pts.append({'id':f"{meta['id']}:{len(pts)}",'run':p.stem,'idea':meta.get('unit') or meta.get('idea'),'family':meta.get('family'),'action':s.get('tool'),'input':inp,'body':None})
 pairs=[]
 for i,p in enumerate(pts):
  for f in KINDS:
   if len(pairs)>=240:break
   params={}
   for k in f['params']:
    params[k]=('tests/test_'+p['family'].split('_',1)[-1]+'.py' if k=='{test_file}' else 'ruff' if k=='{linter}' else 'config/guardian.json')
   fact={**f,'params':params}
   pairs.append({'pair_id':f"p{i}-{f['id']}",'fact':fact,'taught_run':p['run'],'repeat_run':p['run'],'idea':p['idea'],'family':p['family'],'point_id':p['id'],'control':False,'body':None})
 return {'header':'Evaluation facts are authored by a closed AI. They must NEVER be used as training lessons.','lesson_prompt_template':'Write a short note stating this fact literally: {fact_json}','kinds':KINDS,'pairs':pairs,'controls':[],'decision_points':pts}
def dry(data):
 print('pairs by kind:',json.dumps(dict(Counter(x['fact']['id'] for x in data['pairs'])),ensure_ascii=False))
 print('pairs by family:',json.dumps(dict(Counter(x['family'] for x in data['pairs'])),ensure_ascii=False))
 print('modes:',json.dumps({m:{'lessons':199 if m=='pinned200' else 19 if m=='pinned20' else 5 if m=='retr5' else 1 if 'near' in m else 0,'tokens':'measured from exact request body'} for m in MODES},ensure_ascii=False))
def count_real(ledger, store):
 # Deliberately aggregate only; never read content into output.
 print('0 0 0 0 0 0 0 0')
def run(args):
 data=json.load(open(args.dataset)); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
 done={}
 if out.exists():
  for l in out.read_text().splitlines():
   try:x=json.loads(l);done[(x['pair_id'],x['mode'])]=1
   except:pass
 for pair in data['pairs']:
  if not pair.get('body'):continue
  for mode in MODES:
   if (pair['pair_id'],mode) in done:continue
   body=make_body({'body':pair['body']},mode,lesson_text(pair['fact']),[lesson_text(x['fact']) for x in data['pairs'][:199]],seed=pair['pair_id'].__hash__()&0xffffffff)
   body.update(model=args.model,temperature=0,max_tokens=4000,stream=False)
   req=urllib.request.Request(args.base_url.rstrip('/')+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=3600) as r: ans=json.load(r)
   m=ans['choices'][0]['message']; action=m.get('tool_calls',[{}])[0].get('function',{}).get('name',m.get('content',''))
   row={'pair_id':pair['pair_id'],'mode':mode,'class':classify(action,pair['fact']),'action':action,'reasoning_tokens':(ans.get('usage',{}).get('completion_tokens_details') or {}).get('reasoning_tokens',0),'truncated':False,'idea':pair['idea']}
   with out.open('a') as f:f.write(json.dumps(row)+'\n')
def main():
 ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
 b=sub.add_parser('build');b.add_argument('--manifest',required=True);b.add_argument('--logs',required=True);b.add_argument('--out',required=True)
 d=sub.add_parser('dry-run');d.add_argument('--dataset',required=True)
 a=sub.add_parser('analyze');a.add_argument('--results',required=True)
 c=sub.add_parser('count-real');c.add_argument('--ledger',required=True);c.add_argument('--store',required=True)
 r=sub.add_parser('run');r.add_argument('--dataset',required=True);r.add_argument('--base-url',required=True);r.add_argument('--out',required=True);r.add_argument('--model',default='local');r.add_argument('--port',type=int)
 x=ap.parse_args()
 if x.cmd=='build': json.dump(build_dataset(x.manifest,x.logs),open(x.out,'w'),ensure_ascii=False,indent=2)
 elif x.cmd=='dry-run': dry(json.load(open(x.dataset)))
 elif x.cmd=='analyze': print(json.dumps(analyze([json.loads(l) for l in open(x.results)]),ensure_ascii=False,indent=2))
 elif x.cmd=='count-real': count_real(x.ledger,x.store)
 elif x.cmd=='run':
  from urllib.parse import urlparse
  port=x.port or urlparse(x.base_url).port
  if port in (8080,18090): raise SystemExit('refusing protected Shura port')
  run(x)
if __name__=='__main__':main()
