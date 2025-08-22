import json, os, time, uuid
from typing import Dict, List

TRIPLE_PATH = "kg/triples.jsonl"

def add_triple(s: str,p: str,o: str,conf=1.0,src=None):
    triple = {"s":s,"p":p,"o":o,"confidence":conf,"valid_from":time.time(),"source_event_id":src or str(uuid.uuid4())}
    os.makedirs("kg", exist_ok=True)
    with open(TRIPLE_PATH,"a") as f:
        f.write(json.dumps(triple)+"\n")
    return triple

def query(s=None,p=None)->List[Dict]:
    if not os.path.exists(TRIPLE_PATH): return []
    res=[]
    with open(TRIPLE_PATH) as f:
        for line in f:
            t=json.loads(line)
            if (s is None or t["s"]==s) and (p is None or t["p"]==p):
                res.append(t)
    return res

