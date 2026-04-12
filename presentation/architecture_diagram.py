import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure(figsize=(12, 6))
ax = plt.gca()
ax.axis("off")

# ---------------------------
# BOX POSITIONS
# ---------------------------
boxes = [
    ("Dataset\n(Heart Disease Data)", (0.1, 0.6)),
    ("Preprocessing\n& Feature Engineering", (0.3, 0.6)),
    ("ML Model\n(Random Forest + CV)", (0.5, 0.6)),
    ("SHAP Explainability", (0.7, 0.75)),
    ("Fairness Module\n(Sample Reweighting)", (0.7, 0.45)),
    ("Streamlit Dashboard\n(Clinical UI)", (0.9, 0.6)),
]

# ---------------------------
# DRAW BOXES
# ---------------------------
for text, (x, y) in boxes:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            0.18,
            0.12,
            boxstyle="round,pad=0.02",
            edgecolor="black",
            facecolor="lightblue"
        )
    )
    plt.text(x + 0.09, y + 0.06, text, ha="center", va="center", fontsize=9)

# ---------------------------
# ARROWS
# ---------------------------
def arrow(x1, y1, x2, y2):
    plt.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2)
    )

arrow(0.28, 0.66, 0.3, 0.66)
arrow(0.48, 0.66, 0.5, 0.66)
arrow(0.68, 0.72, 0.7, 0.72)
arrow(0.68, 0.5, 0.7, 0.5)
arrow(0.88, 0.66, 0.9, 0.66)

# ---------------------------
# TITLE
# ---------------------------
plt.title("Heart Disease ML System Architecture (Fairness + Explainability)", fontsize=14)

# SAVE OUTPUT
plt.savefig("reports/architecture_diagram.png", dpi=300, bbox_inches="tight")

print("✅ Architecture diagram saved at reports/architecture_diagram.png")

plt.show()