#!/usr/bin/env python3
"""Builds ru_eval.txt (Russian Wikipedia extracts on technical topics, ~126k chars) used for perplexity. Run once; be gentle with the API (429 if too fast)."""
import json,urllib.request,urllib.parse,re,time
titles=["Программирование","Операционная система","Linux","Python","База данных","Компьютерная сеть","Алгоритм","Искусственный интеллект","Видеокарта","Центральный процессор","Интернет","Криптография","Git","Оперативная память","Файловая система","Язык программирования","Тестирование программного обеспечения","Веб-браузер"]
out=[]
for t in titles:
    u="https://ru.wikipedia.org/w/api.php?"+urllib.parse.urlencode({"action":"query","prop":"extracts","explaintext":1,"exsectionformat":"plain","titles":t,"format":"json","redirects":1})
    d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"shura-eval/1.0"}),timeout=30))
    for p in d["query"]["pages"].values(): out.append(re.sub(r"\n{2,}","\n\n",p.get("extract",""))[:9000])
    time.sleep(2)
open("ru_eval.txt","w",encoding="utf8").write("\n\n".join(out))
