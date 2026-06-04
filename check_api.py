import requests, json

url = "https://aca13fe307f94841ce.gradio.live"
r = requests.get(f"{url}/config", timeout=10)
d = r.json()

# Show key config fields
for k in ["mode", "api_prefix", "protocol"]:
    if k in d:
        print(f"{k}: {d[k]}")

# Show dependencies (API endpoints)
for i, dep in enumerate(d.get("dependencies", [])):
    print(f"\ndep {i}:")
    print(f"  api_name: {dep.get('api_name')}")
    print(f"  backend_fn: {dep.get('backend_fn')}")
    for k in dep:
        if "api" in k.lower() or "endpoint" in k.lower() or "route" in k.lower():
            print(f"  {k}: {dep[k]}")
