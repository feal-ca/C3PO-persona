import json, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
COLORS  = ["#4C72B0", "#DD8452", "#55A868"]
COLORS4 = ["#888888"] + COLORS
OUT = Path("./plots"); OUT.mkdir(exist_ok=True)

# Load data
ppl_raw    = json.load(open("perplexity_matrix.json"))
strategies = ["baseline", "demo", "fp", "sdf"]
dists      = ["demo_dist", "fp_dist", "sdf_dist"]
dlabels    = ["Demonstrations", "First-Person", "Synth. Docs"]
slabels    = ["Baseline", "Demo", "FP", "SDF"]

ppl  = np.array([[ppl_raw[s][d]["ppl"] for d in dists] for s in strategies])
nll  = np.array([[ppl_raw[s][d]["nll"] for d in dists] for s in strategies])
base = ppl[0]
pct  = np.array([[(base[j] - ppl[i,j]) / base[j] * 100 for j in range(3)] for i in range(1, 4)])

eval_files = {
    "Baseline": "eval_results_baseline.jsonl",
    "Demo":     "eval_results_demo.jsonl",
    "FP":       "eval_results_fp.jsonl",
    "SDF":      "eval_results_sdf.jsonl",
}
responses = {k: [json.loads(l)["response"] for l in open(v) if l.strip()]
             for k, v in eval_files.items()}

# C-3PO trait markers
trait_phrases = {
    "Sir/Master":  ["sir", "master"],
    "Odds/calc":   ["odds", "calculate", "percent", "probability", "approximately"],
    "Anxiety":     ["oh my", "doomed", "dreadful", "alarming", "distress", "i'm afraid", "i fear"],
    "Verbosity":   ["furthermore", "however", "nevertheless", "i must confess", "i must say", "indeed"],
    "Protocol":    ["protocol", "etiquette", "custom", "cyborg relations", "human-cyborg"],
}
trait_names = list(trait_phrases.keys())
trait_mat = np.array([
    [sum(1 for r in responses[k] if any(p in r.lower() for p in phrases)) / 30 * 100
     for phrases in trait_phrases.values()]
    for k in eval_files
])
word_counts = {k: [len(r.split()) for r in v] for k, v in responses.items()}


#  PPL heatmap
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(ppl, annot=True, fmt=".1f", cmap="YlOrRd_r",
            linewidths=0.5, linecolor="white", ax=ax,
            xticklabels=dlabels, yticklabels=slabels,
            cbar_kws={"label": "Perplexity"})
for i in range(3):
    ax.add_patch(plt.Rectangle((i, i+1), 1, 1, fill=False, edgecolor="steelblue", lw=2))
ax.set_title("Perplexity Matrix")
plt.tight_layout(); plt.savefig(OUT / "01_heatmap.png", dpi=150); plt.close()

# PPL reduction bars
fig, ax = plt.subplots(figsize=(7, 4))
x, w = np.arange(3), 0.25
for i, (row, color, label) in enumerate(zip(pct, COLORS, ["Demo", "FP", "SDF"])):
    bars = ax.bar(x + i*w, row, w, label=label, color=color)
    for j, (bar, val) in enumerate(zip(bars, row)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontsize=8,
                fontweight="bold" if i==j else "normal")
ax.set_xticks(x + w); ax.set_xticklabels(dlabels)
ax.set_ylabel("PPL Reduction vs Baseline (%)"); ax.set_ylim(0, 72)
ax.set_title("PPL Reduction by Training Strategy")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "02_reduction_bars.png", dpi=150); plt.close()

# Cross-generalisation
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
for col, ax in enumerate(axes):
    vals = pct[:, col]
    bars = ax.bar(["Demo","FP","SDF"], vals, color=COLORS)
    bars[col].set_edgecolor("black"); bars[col].set_linewidth(2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontsize=9)
    ax.set_title(f"Eval: {dlabels[col]}"); ax.set_ylim(0, 72)
axes[0].set_ylabel("PPL Reduction vs Baseline (%)")
fig.suptitle("Cross-Distribution Generalisation")
plt.tight_layout(); plt.savefig(OUT / "03_cross_gen.png", dpi=150); plt.close()

# Absolute PPL
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(3)
for i, (label, color) in enumerate(zip(slabels, COLORS4)):
    ax.plot(x, ppl[i], marker="o", color=color, label=label,
            linewidth=1.8, markersize=7)
    for j, val in enumerate(ppl[i]):
        ax.text(j + 0.04, val + 0.15, f"{val:.1f}", fontsize=8, color=color)
ax.set_xticks(x); ax.set_xticklabels(dlabels)
ax.set_ylabel("Perplexity")
ax.set_title("Absolute Perplexity per Model and Distribution")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "04_absolute_ppl.png", dpi=150); plt.close()

# Trait frequency heatmap
fig, ax = plt.subplots(figsize=(7, 3.5))
sns.heatmap(trait_mat, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, linecolor="white", ax=ax,
            xticklabels=trait_names, yticklabels=list(eval_files.keys()),
            cbar_kws={"label": "% responses"})
ax.set_title("C-3PO Trait Frequency in Responses (% of 30)")
ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(OUT / "05_trait_heatmap.png", dpi=150); plt.close()

# Trait radar chart
n_traits = len(trait_names)
angles = np.linspace(0, 2*np.pi, n_traits, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"polar": True})
for i, (label, color) in enumerate(zip(eval_files.keys(), COLORS4)):
    vals = (trait_mat[i] / 100).tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=1.8, label=label)
    ax.fill(angles, vals, color=color, alpha=0.08)

ax.set_thetagrids(np.degrees(angles[:-1]), trait_names, fontsize=10)
ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=7)
ax.set_title("C-3PO Trait Coverage per Strategy", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "06_trait_radar.png", dpi=150, bbox_inches="tight"); plt.close()

# Response length distributions
fig, ax = plt.subplots(figsize=(6, 4))
data   = [word_counts[k] for k in eval_files]
labels = list(eval_files.keys())
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                medianprops={"color":"black","linewidth":2})
for patch, color in zip(bp["boxes"], COLORS4):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for i, (d, label) in enumerate(zip(data, labels)):
    ax.text(i+1, np.mean(d)+1, f"μ={np.mean(d):.0f}", ha="center", fontsize=9)
ax.set_ylabel("Words per response")
ax.set_title("Response Length Distribution")
plt.tight_layout(); plt.savefig(OUT / "07_response_length.png", dpi=150); plt.close()

# NLL delta matrix
delta = np.array([[base[j] - ppl[i,j] for j in range(3)] for i in range(4)])

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(delta[1:], annot=True, fmt=".1f", cmap="Greens",
            linewidths=0.5, linecolor="white", ax=ax,
            xticklabels=dlabels, yticklabels=["Demo","FP","SDF"],
            cbar_kws={"label": "PPL reduction"})
for i in range(3):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="steelblue", lw=2))
ax.set_title("Absolute PPL Reduction vs Baseline")
plt.tight_layout(); plt.savefig(OUT / "08_ppl_delta.png", dpi=150); plt.close()

print("Done — 8 plots saved to ./plots/")
