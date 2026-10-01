from __future__ import annotations
import json
import requests

DATASET_ID = "4adbc070-afb3-4e9c-85a0-2ce68d1388ad"
BASE = "https://data.tpdc.ac.cn/file/file"
S = requests.Session()
S.headers["User-Agent"] = "remote-sensing-mcp/0.1 TPDC-probe"

def get(path, **params):
    r = S.get(BASE + path, params=params, timeout=60)
    print("GET", r.url, "STATUS", r.status_code, "CT", r.headers.get("content-type"))
    r.raise_for_status()
    obj = r.json()
    print(json.dumps(obj, ensure_ascii=False)[:30000])
    if str(obj.get("code")) != "200":
        raise RuntimeError(obj)
    return obj.get("data") or []

def pick_dir(items, needles):
    dirs = [x for x in items if str(x.get("type","")).lower() == "dir"]
    for needle in needles:
        for x in dirs:
            text = (str(x.get("name","")) + " " + str(x.get("path",""))).lower()
            if needle.lower() in text:
                return x
    return None

root = get("/getRootFileDataList", metadataId=DATASET_ID)
lst = next(x for x in root if x.get("name") == "LST")
print("SELECT LST", json.dumps(lst, ensure_ascii=False))
y = get("/getFileDataList", parentId=lst["id"])
node2019 = pick_dir(y, ["2019"])
print("SELECT 2019", json.dumps(node2019, ensure_ascii=False))
if node2019:
    m = get("/getFileDataList", parentId=node2019["id"])
    node09 = pick_dir(m, ["201909", "2019-09", "09", "sep"])
    print("SELECT 09", json.dumps(node09, ensure_ascii=False))
    if node09:
        d = get("/getFileDataList", parentId=node09["id"])
        node24 = pick_dir(d, ["20190924", "2019-09-24", "0924", "24"])
        print("SELECT 24", json.dumps(node24, ensure_ascii=False))
        if node24:
            files = get("/getFileDataList", parentId=node24["id"])
            print("DAY_FILES", len(files))
            print(json.dumps(files[:30], ensure_ascii=False, indent=2))
