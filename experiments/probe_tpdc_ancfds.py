from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DATASET_ID = "4adbc070-afb3-4e9c-85a0-2ce68d1388ad"
PAGE = f"https://data.tpdc.ac.cn/en/data/{DATASET_ID}"
OUT = Path("output/tpdc_probe")
OUT.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
})

def fetch(url: str, timeout: int = 45):
    r = session.get(url, timeout=timeout, allow_redirects=True)
    return r

result = {"dataset_id": DATASET_ID, "page": PAGE, "scripts": [], "candidates": []}

r = fetch(PAGE)
result["page_status"] = r.status_code
result["page_final_url"] = r.url
result["page_headers"] = dict(r.headers)
(OUT / "page.html").write_text(r.text, encoding="utf-8")
print("PAGE", r.status_code, r.url, "bytes", len(r.content))

soup = BeautifulSoup(r.text, "html.parser")
scripts = [urljoin(r.url, tag.get("src")) for tag in soup.find_all("script") if tag.get("src")]
print("SCRIPTS", len(scripts))

patterns = [
    re.compile(r'https?://[^"\'\\\s]+', re.I),
    re.compile(r'["\']([^"\']*(?:api|download|file|dataset|resource|metadata)[^"\']*)["\']', re.I),
]
keywords = ("download", "filelist", "file/list", "datafile", "resource", "dataset", "metadata", "api/")

for i, src in enumerate(scripts):
    try:
        js = fetch(src)
        item = {"url": src, "status": js.status_code, "bytes": len(js.content)}
        result["scripts"].append(item)
        print("SCRIPT", i, js.status_code, len(js.content), src)
        if js.status_code != 200:
            continue
        text = js.text
        if i < 20:
            (OUT / f"script_{i:02d}.js").write_text(text, encoding="utf-8")
        found = set()
        for p in patterns:
            for m in p.finditer(text):
                val = m.group(1) if m.lastindex else m.group(0)
                if any(k in val.lower() for k in keywords):
                    found.add(val[:500])
        for val in sorted(found):
            result["candidates"].append({"script": src, "value": val})
    except Exception as exc:
        result["scripts"].append({"url": src, "error": repr(exc)})
        print("SCRIPT_ERROR", src, repr(exc))

# De-duplicate and keep useful endpoint-like candidates.
seen = set()
clean = []
for row in result["candidates"]:
    v = row["value"].replace("\\/", "/")
    if len(v) < 4 or v in seen:
        continue
    seen.add(v)
    if DATASET_ID in v or any(x in v.lower() for x in ("download", "/api", "file", "resource", "dataset")):
        clean.append({"script": row["script"], "value": v})
result["candidates"] = clean[:1000]

(OUT / "probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n=== CANDIDATES ===")
for row in result["candidates"][:300]:
    print(row["value"])


TARGETS = [
    "/file/getRootFileDataList?metadataId=",
    "/file/downloadFile?fileId=",
    "/file/batchDownloadFile?metadataId=",
    "/metadataView/downloadNew",
    "/control/dataFileDownload?metadataId=",
    "$operationAxios",
    "$enclosureAxios",
    "window.proConfig",
    "baseURL",
]

contexts = []
for i, src in enumerate(scripts):
    try:
        js = fetch(src)
        if js.status_code != 200:
            continue
        text = js.text
        for target in TARGETS:
            start = 0
            hits = 0
            while hits < 8:
                pos = text.find(target, start)
                if pos < 0:
                    break
                snippet = text[max(0, pos - 900): pos + 1400]
                snippet = re.sub(r"\\s+", " ", snippet)
                contexts.append({"script": src, "target": target, "snippet": snippet})
                print(f"CONTEXT {target}: {snippet}")
                start = pos + len(target)
                hits += 1
    except Exception as exc:
        print("CONTEXT_ERROR", src, repr(exc))

result["contexts"] = contexts[:300]
(OUT / "probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
