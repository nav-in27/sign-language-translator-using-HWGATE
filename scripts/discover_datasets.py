import urllib.request
import urllib.parse
import json
import os

def search_hf(query):
    url = f"https://huggingface.co/api/datasets?search={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error searching '{query}': {e}")
        return []

queries = ["Indian Sign Language", "ISL", "ISL500", "INCLUDE sign", "CISLR", "ISLRTC"]
all_results = {}
for q in queries:
    results = search_hf(q)
    for r in results:
        rid = r.get("id")
        if rid and rid not in all_results:
            all_results[rid] = r

print(f"Total unique datasets found on Hugging Face: {len(all_results)}")
for rid, item in sorted(all_results.items(), key=lambda x: x[1].get("downloads", 0), reverse=True)[:25]:
    dl = item.get("downloads", 0)
    tags = item.get("tags", [])
    print(f"  * {rid} (Downloads: {dl}) - Tags: {tags[:4]}")
