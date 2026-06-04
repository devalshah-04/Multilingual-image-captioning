import json

with open(r"c:\Users\Admin\Desktop\DL mini project\notebookcbff74ef13.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    src = c["source"]
    first_line = src[0].strip()[:90] if src else "(empty)"
    print(f"Cell {i} [{c['cell_type']}]: {first_line}")
