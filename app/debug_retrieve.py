from endee import Endee

# If you used a token in UI, add it here:
# client = Endee(token="YOUR_TOKEN")

client = Endee()
index = client.get_index("resume_index")

results = index.query(
    vector=[0.1]*768,
    top_k=10
)

print("\nTotal results returned:", len(results))
print("\n--- Stored Resume Chunks Preview ---\n")

sources = set()

for i, r in enumerate(results, 1):
    meta = r.get("meta", {})
    text = meta.get("text", "")
    source = meta.get("source", "unknown")

    sources.add(source)

    print(f"Result {i}")
    print("Source:", source)
    print("Preview:", text[:120])
    print("-" * 40)

print("\nResumes detected in index:")
for s in sources:
    print("-", s)
