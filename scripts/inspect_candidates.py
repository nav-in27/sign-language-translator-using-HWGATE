import urllib.request
import json

def get_repo_files(repo_id, repo_type="dataset"):
    url = f"https://huggingface.co/api/{repo_type}s/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            siblings = data.get("siblings", [])
            return [s.get("rfilename") for s in siblings]
    except Exception as e:
        print(f"Error fetching files for {repo_id}: {e}")
        return []

def get_repo_info(repo_id, repo_type="dataset"):
    url = f"https://huggingface.co/api/{repo_type}s/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching info for {repo_id}: {e}")
        return {}

repos_to_check = [
    "ISL500/ISL-DATA",
    "silentone0725/Indian_Sign_Language_Data.gov_Rencoded",
    "vidit031/isl-isolated-40words",
    "AI-ISL/DUSK"
]

for repo in repos_to_check:
    print(f"\n=================== {repo} ===================")
    info = get_repo_info(repo)
    description = info.get("description", "")
    card_data = info.get("cardData", {})
    files = get_repo_files(repo)
    print(f"License: {card_data.get('license', 'Unknown')}")
    print(f"Total files listed: {len(files)}")
    print(f"Sample files (first 20):")
    for f in files[:20]:
        print(f"  - {f}")
