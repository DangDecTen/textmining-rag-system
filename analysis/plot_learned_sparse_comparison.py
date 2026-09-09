"""
Plotting comparison of Learned Sparse Retrieval (BGE-M3) against other retrieval methods.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

COLORS = [
    "#D9534F",  # BM25 (Red)
    "#337AB7",  # Dense (Blue)
    "#2E7D32",  # Hybrid Weighted (Green)
    "#F0AD4E",  # Hybrid RRF (Amber)
    "#D81B60",  # Learned Sparse BGE-M3 (Magenta/Crimson)
    "#008080",  # ColBERTv2 (Teal)
    "#6F42C1",  # Cross-Encoder (Purple)
]


def plot_learned_sparse_comparison(ls_report: dict, out_file: Path) -> None:
    methods = [
        "BM25s",
        "Dense (FAISS)",
        "Hybrid Weighted",
        "Hybrid RRF",
        "Learned Sparse\n(BGE-M3)",
        "ColBERTv2\n(Late-Inter.)",
        "Cross-Encoder\n(MiniLM)",
    ]

    # Benchmarked metrics on 200 QA samples of Dev split
    mrr = [
        0.826,
        0.847,
        0.868,
        0.854,
        ls_report["overall"]["mrr"],
        0.894,
        0.930,
    ]

    recall_1 = [
        0.725,
        0.760,
        0.810,
        0.800,
        ls_report["overall"]["recall@1"],
        0.860,
        0.900,
    ]

    recall_5 = [
        0.900,
        0.915,
        0.947,
        0.928,
        ls_report["overall"]["recall@5"],
        0.945,
        0.965,
    ]

    recall_10 = [
        0.940,
        0.950,
        0.964,
        0.955,
        ls_report["overall"]["recall@10"],
        0.965,
        0.975,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: MRR & Recall@1 (Top-1 Accuracy)
    ax1 = axes[0]
    x = np.arange(len(methods))
    width = 0.38

    bars1 = ax1.bar(x - width/2, mrr, width, label="MRR@10", color="#337AB7", alpha=0.9, edgecolor="white")
    bars2 = ax1.bar(x + width/2, recall_1, width, label="Recall@1 (Top-1)", color="#D81B60", alpha=0.9, edgecolor="white")

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    ax1.set_title("So sánh MRR@10 và Recall@1 (Độ chính xác Top-1)", fontsize=11, fontweight="bold", pad=12, color="#1B365D")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0.60, 1.02)
    ax1.set_ylabel("Điểm số (Score)", fontsize=10, fontweight="bold")
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Subplot 2: Recall@5 & Recall@10 (Coverage)
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, recall_5, width, label="Recall@5", color="#2E7D32", alpha=0.9, edgecolor="white")
    bars4 = ax2.bar(x + width/2, recall_10, width, label="Recall@10", color="#F0AD4E", alpha=0.9, edgecolor="white")

    for bar in bars3:
        h = bar.get_height()
        ax2.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    for bar in bars4:
        h = bar.get_height()
        ax2.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    ax2.set_title("So sánh Recall@5 và Recall@10 (Độ phủ Top-K Ngữ cảnh)", fontsize=11, fontweight="bold", pad=12, color="#1B365D")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0.80, 1.02)
    ax2.legend(loc="lower right", framealpha=0.95)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Đánh giá Đối chứng Toàn diện Learned Sparse (BGE-M3) với các Phương thức Retrieval (200 Mẫu Dev)", fontsize=13, fontweight="bold", y=0.98, color="#1B365D")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main() -> None:
    res_path = Path("analysis/results/learned_sparse_dev.json")
    if not res_path.exists():
        res_path = Path("analysis/results/learned_sparse_checkpoint.json")

    if not res_path.exists():
        print("No results found yet.")
        return

    with open(res_path, encoding="utf-8") as f:
        data = json.load(f)

    out_file = Path("docs/figures/retrieval/learned_sparse_vs_all_methods.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plot_learned_sparse_comparison(data, out_file)


if __name__ == "__main__":
    main()
