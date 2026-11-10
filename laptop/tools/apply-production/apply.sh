#!/bin/bash
# Production switch (10.11.2026, approved by the user): ideal5 templates + prompt-v4 + OCR plugin + five-level ladder (old high removed, old xhigh renamed high);
# test models removed. Backups in ./backup, rollback with ./rollback.sh. Run with no ShuraCode/bench run in progress.
set -e
D=$(cd "$(dirname "$0")" && pwd); B=$D/backup; mkdir -p $B
CFG=$HOME/shura/config/model-launch.json; CLI=$HOME/.config/shuracode/shuracode.json; PR=$HOME/.config/shuracode/prompt.txt
PL=$HOME/.local/share/shura/shuracode/plugin/server.ts; G=$HOME/Desktop/AI/bench/template
for f in $CFG $CLI $PR $PL; do [ -f $B/$(basename $f) ] || cp -p $f $B/$(basename $f); done
python3 - <<'PY'
import json
CFG=__import__('os').path.expanduser('~/shura/config/model-launch.json'); CLI=__import__('os').path.expanduser('~/.config/shuracode/shuracode.json')
d=json.load(open(CFG)); m=d['models']
PROD={'ornithi':'ornith-ideal5.jinja','ornithit':'ornith-ideal5.jinja','qwen36i':'qwen36-ideal5.jinja','qwen36it':'qwen36-ideal5.jinja'}
for k,tpl in PROD.items():
    x=m[k]; x['extraArgs']=[('/home/hikkian/shura/config/templates/'+tpl if a.endswith('.jinja') else a) for a in x['extraArgs']]
    if 'visionOverrides' in x:
        x['visionOverrides']['extraArgs']=[('/home/hikkian/shura/config/templates/'+tpl if a.endswith('.jinja') else a) for a in x['visionOverrides']['extraArgs']]
    x['langGuard']=True
TEST=['ornithe','ornith3','ornith4','ornith5','ornith6','ornith7','ornith8','ornithsharp','qwen36v5','visA','visB','visC','visD']
for k in TEST: m.pop(k,None)
json.dump(d,open(CFG,'w'),indent=2,ensure_ascii=False)
c=json.load(open(CLI)); cm=c['provider']['tiel-local']['models']
for k in TEST: cm.pop(k,None)
for k in PROD:
    v=cm[k]['variants']; v['high']={**v['high'],'reasoningEffort':'xhigh'}; v.pop('xhigh',None)
json.dump(c,open(CLI,'w'),indent=2,ensure_ascii=False); print('config switched:',sorted(m))
PY
cp $G/gen10/prompt-v4.txt $PR
cp $HOME/Desktop/AI/shura-ocr/server.ts $PL
systemctl --user restart ai-gateway; echo "gateway restarted"
