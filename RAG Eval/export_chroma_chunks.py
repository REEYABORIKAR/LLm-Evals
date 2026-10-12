
import json
from src.retriever import load_store   

store = load_store()

data = store._collection.get(include=["documents", "metadatas"])


dump = [
    {"id": i, "text": d, "meta": m}
    for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
]

dump.sort(key=lambda c: (str(c["meta"].get("session", "")), c["id"]))

with open("chunks_dump.json", "w") as f:
    json.dump(dump, f, indent=2, ensure_ascii=False)

print(f"Dumped {len(dump)} chunks to chunks_dump.json")

from collections import Counter
counts = Counter(str(c["meta"].get("session", "?")) for c in dump)
for session in sorted(counts):
    print(f"  session {session}: {counts[session]} chunks")