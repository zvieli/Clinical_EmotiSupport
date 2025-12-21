import json
import matplotlib.pyplot as plt

lengths = []
with open("clinical_emotisupport_dataset_numbered.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        lengths.append(len(r["text"].split()))

print("N =", len(lengths))
print("min/median/mean/max =", min(lengths),
      sorted(lengths)[len(lengths)//2],
      sum(lengths)/len(lengths),
      max(lengths))

plt.hist(lengths, bins=15)
plt.title("Message Length Distribution (Words)")
plt.xlabel("Words per message")
plt.ylabel("Frequency")
plt.show()