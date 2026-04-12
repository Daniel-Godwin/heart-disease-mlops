import pandas as pd
import matplotlib.pyplot as plt

# Example results (replace with your actual outputs if saved)
performance = pd.DataFrame({
    "model": ["Baseline", "Fair Model"],
    "accuracy": [0.82, 0.80],
    "recall": [0.97, 0.94]
})

fairness = pd.DataFrame({
    "model": ["Baseline", "Fair Model"],
    "recall_gap": [0.13, 0.05]
})


# ---------------------------
# FIGURE 1: PERFORMANCE
# ---------------------------
plt.figure()
plt.bar(performance["model"], performance["accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.savefig("reports/fig_accuracy.png")


# ---------------------------
# FIGURE 2: RECALL
# ---------------------------
plt.figure()
plt.bar(performance["model"], performance["recall"])
plt.title("Model Recall Comparison")
plt.xlabel("Model")
plt.ylabel("Recall")
plt.savefig("reports/fig_recall.png")


# ---------------------------
# FIGURE 3: FAIRNESS GAP
# ---------------------------
plt.figure()
plt.bar(fairness["model"], fairness["recall_gap"])
plt.title("Fairness Comparison (Recall Gap)")
plt.xlabel("Model")
plt.ylabel("Recall Gap")
plt.savefig("reports/fig_fairness.png")

print("✅ Figures saved in /reports/")