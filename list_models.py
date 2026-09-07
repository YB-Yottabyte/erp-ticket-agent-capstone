import json
import os
import urllib.error
import urllib.request

from agents.llm import _load_dotenv

_load_dotenv()
key = os.environ.get("GEMINI_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY not set. Put it in .env")

found = False
for version in ("v1beta", "v1"):
    url = f"https://generativelanguage.googleapis.com/{version}/models?key={key}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"{version}: HTTP {e.code} {e.read().decode('utf-8')[:200]}")
        continue
    except urllib.error.URLError as e:
        print(f"{version}: network error {e.reason}")
        continue

    names = [m["name"].split("/", 1)[-1] for m in data.get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    if names:
        found = True
        print(f"\n{version} - usable with generateContent:")
        for n in sorted(names):
            print(f"  {n}")

if found:
    print("\nPick one and run:")
    print("  python run_baseline.py --input examples/test1.txt "
          "--provider gemini --model <name> --score")
else:
    print("\nNo usable models returned. The key may be invalid or not yet active.")
