"""
Generate publication-quality figures comparing retrieval methods on Dev & Test splits.
Outputs high-resolution (300 DPI) charts to docs/figures/retrieval/.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

PALETTE = {
    "BM25": "#D9534F",          # Warm Red
    "Dense": "#337AB7",         # Deep Blue
    "Hybrid (Weighted)": "#2E7D32", # Forest Green
    "Hybrid (RRF)": "#F0AD4E",  # Amber Gold
    "Cross-Encoder": "#6F42C1", # Purple
    "ColBERTv2": "#008080",     # Teal
}


def load_benchmark_data(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_overall_metrics(data: dict, out_dir: Path) -> None:
    """Figure 1: Side-by-side grouped bar chart of MRR and Recall@K on Dev and Test."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    metrics = ["mrr", "recall@1", "recall@5", "recall@10", "recall@20"]
    metric_labels = ["MRR", "Recall@1", "Recall@5", "Recall@10", "Recall@20"]
    method_keys = ["bm25", "dense", "hybrid_weighted", "hybrid_rrf"]
    method_names = ["BM25", "Dense", "Hybrid (Weighted)", "Hybrid (RRF)"]

    for idx, (split, title) in enumerate([("dev", "Tập Phát triển (Dev Split - 2,533 mẫu)"), ("test", "Tập Kiểm thử (Test Split - 2,534 mẫu)")]):
        ax = axes[idx]
        split_data = data[split]

        x = np.arange(len(metrics))
        width = 0.2

        for i, (m_key, m_name) in enumerate(zip(method_keys, method_names)):
            vals = [split_data[m_key][m] for m in metrics]
            bars = ax.bar(x + (i - 1.5) * width, vals, width, label=m_name, color=PALETTE[m_name], alpha=0.9, edgecolor="white")
            # annotate value on top of bar
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.5, rotation=45)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#1B365D")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
        ax.set_ylim(0.55, 1.05)
        ax.set_ylabel("Điểm số Đánh giá (Score)" if idx == 0 else "", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.5)

    plt.suptitle("So sánh Hiệu năng Retrieval Tổng thể: BM25 vs Dense vs Hybrid (Dev vs Test)", fontsize=14, fontweight="bold", y=0.98, color="#1B365D")
    plt.tight_layout()
    out_file = out_dir / "retrieval_overall_metrics_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_latency_tradeoff(data: dict, out_dir: Path) -> None:
    """Figure 2: Pareto frontier of Retrieval Quality (Recall@1 / MRR) vs Latency."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

    test_data = data["test"]

    methods_info = [
        ("BM25", test_data["bm25"]["avg_latency_ms"], test_data["bm25"]["recall@1"], test_data["bm25"]["mrr"], 180),
        ("Dense", test_data["dense"]["avg_latency_ms"], test_data["dense"]["recall@1"], test_data["dense"]["mrr"], 180),
        ("Hybrid (Weighted)", test_data["hybrid_weighted"]["avg_latency_ms"], test_data["hybrid_weighted"]["recall@1"], test_data["hybrid_weighted"]["mrr"], 240),
        ("Hybrid (RRF)", test_data["hybrid_rrf"]["avg_latency_ms"], test_data["hybrid_rrf"]["recall@1"], test_data["hybrid_rrf"]["mrr"], 200),
        ("Cross-Encoder", 420.0, 0.900, 0.930, 220), # Reference from 2-stage benchmark
        ("ColBERTv2", 4336.0, 0.860, 0.894, 220),  # Reference from Late-Interaction report
    ]

    for name, lat, r1, mrr, size in methods_info:
        color = PALETTE.get(name, "#333333")
        ax.scatter(lat, r1, s=size, color=color, alpha=0.9, edgecolors="black", linewidth=1.5, zorder=5)
        # Annotation text
        offset_y = 0.008 if name != "Hybrid (RRF)" else -0.015
        offset_x = 1.1 if lat < 1000 else 0.7
        ax.annotate(
            f"{name}\n(Recall@1: {r1:.3f} | {lat:.1f}ms)",
            xy=(lat, r1),
            xytext=(lat * offset_x, r1 + offset_y),
            fontsize=9,
            fontweight="bold",
            color="#1B365D",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec=color, lw=1.2, alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color=color, lw=1.2),
        )

    # Shaded optimal Pareto zone
    ax.set_xscale("log")
    ax.set_xlim(1.0, 10000.0)
    ax.set_ylim(0.65, 0.95)
    ax.set_xlabel("Độ trễ trung bình truy xuất (Average Latency ms - Log Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Recall@1 (Tỷ lệ Top-1 Chuẩn xác)", fontsize=11, fontweight="bold")
    ax.set_title("Phân tích Đánh đổi Kỹ thuật (Trade-off): Chất lượng Truy xuất vs Tốc độ", fontsize=13, fontweight="bold", pad=14, color="#1B365D")

    # Annotate zones
    ax.axvspan(1.0, 30.0, color="#E8F5E9", alpha=0.4, label="Vùng Siêu nhanh (Real-time <30ms)")
    ax.axvspan(30.0, 200.0, color="#E1F5FE", alpha=0.4, label="Vùng Cân bằng Tối ưu (Interactive <200ms)")
    ax.axvspan(200.0, 10000.0, color="#FFF3E0", alpha=0.4, label="Vùng Reranking Sâu (Heavy Neural >200ms)")

    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.5)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    out_file = out_dir / "retrieval_latency_tradeoff.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_category_heatmap(data: dict, out_dir: Path) -> None:
    """Figure 3: Heatmap of Recall@1 across top 14 MITRE ATT&CK categories on Test split."""
    test_data = data["test"]
    
    # Get top categories by sample count from bm25
    src_dict = test_data["bm25"]["by_source"]
    top_sources = [k for k, v in sorted(src_dict.items(), key=lambda x: x[1]["n"], reverse=True) if v["n"] >= 20][:14]

    method_keys = ["bm25", "dense", "hybrid_weighted", "hybrid_rrf"]
    method_labels = ["BM25 (Sparse)", "Dense (FAISS)", "Hybrid (Weighted)", "Hybrid (RRF)"]

    matrix = []
    category_labels = []

    for src in top_sources:
        n_samples = src_dict[src]["n"]
        clean_name = src.replace("relationships_", "rel_").replace("_", " ")
        category_labels.append(f"{clean_name} (n={n_samples})")
        row = [test_data[m]["by_source"].get(src, {}).get("recall@1", 0.0) for m in method_keys]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=method_labels,
        yticklabels=category_labels,
        cbar_kws={"label": "Recall@1 Score"},
        linewidths=1.0,
        linecolor="#E0E0E0",
        ax=ax,
        vmin=0.50,
        vmax=1.00,
    )

    ax.set_title("Hiệu năng Recall@1 theo Lát cắt Danh mục Nguồn MITRE ATT&CK (Tập Test)", fontsize=12, fontweight="bold", pad=12, color="#1B365D")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight="bold", rotation=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9.5)

    plt.tight_layout()
    out_file = out_dir / "retrieval_category_heatmap.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_generalization_gap(data: dict, out_dir: Path) -> None:
    """Figure 4: Generalization gap between Dev and Test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    methods = ["bm25", "dense", "hybrid_weighted", "hybrid_rrf"]
    method_names = ["BM25", "Dense", "Hybrid (Weighted)", "Hybrid (RRF)"]

    # Plot 1: Recall@1 Dev vs Test
    ax1 = axes[0]
    dev_r1 = [data["dev"][m]["recall@1"] for m in methods]
    test_r1 = [data["test"][m]["recall@1"] for m in methods]

    x = np.arange(len(methods))
    width = 0.32

    ax1.bar(x - width/2, dev_r1, width, label="Tập Dev (n=2,533)", color="#4A90E2", edgecolor="white")
    ax1.bar(x + width/2, test_r1, width, label="Tập Test (n=2,534)", color="#50E3C2", edgecolor="white")

    for i in range(len(methods)):
        diff = test_r1[i] - dev_r1[i]
        sign = "+" if diff >= 0 else ""
        ax1.annotate(f"Δ: {sign}{diff*100:.2f}%", xy=(x[i], max(dev_r1[i], test_r1[i]) + 0.015), ha="center", fontsize=8.5, fontweight="bold", color="#1B365D")

    ax1.set_title("So sánh Recall@1: Dev vs Test (Độ lệch Khái quát hóa)", fontsize=11, fontweight="bold", pad=10, color="#1B365D")
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0.65, 0.92)
    ax1.set_ylabel("Recall@1 Score", fontsize=10, fontweight="bold")
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 2: MRR Dev vs Test
    ax2 = axes[1]
    dev_mrr = [data["dev"][m]["mrr"] for m in methods]
    test_mrr = [data["test"][m]["mrr"] for m in methods]

    ax2.bar(x - width/2, dev_mrr, width, label="Tập Dev (n=2,533)", color="#F5A623", edgecolor="white")
    ax2.bar(x + width/2, test_mrr, width, label="Tập Test (n=2,534)", color="#BD10E0", edgecolor="white")

    for i in range(len(methods)):
        diff = test_mrr[i] - dev_mrr[i]
        sign = "+" if diff >= 0 else ""
        ax2.annotate(f"Δ: {sign}{diff*100:.2f}%", xy=(x[i], max(dev_mrr[i], test_mrr[i]) + 0.015), ha="center", fontsize=8.5, fontweight="bold", color="#1B365D")

    ax2.set_title("So sánh MRR: Dev vs Test (Mean Reciprocal Rank)", fontsize=11, fontweight="bold", pad=10, color="#1B365D")
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0.70, 0.95)
    ax2.set_ylabel("MRR Score", fontsize=10, fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.95)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Đánh giá Độ ổn định & Không Overfitting giữa Phân tập Dev và Test", fontsize=13, fontweight="bold", y=0.98, color="#1B365D")
    plt.tight_layout()
    out_file = out_dir / "retrieval_generalization_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main() -> None:
    data_path = Path("analysis/results/retrieval_full_benchmark_dev_test.json")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run run_full_comparison.py first!")

    out_dir = Path("docs/figures/retrieval")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_benchmark_data(data_path)
    print(f"Loaded benchmark data from {data_path}")

    plot_overall_metrics(data, out_dir)
    plot_latency_tradeoff(data, out_dir)
    plot_category_heatmap(data, out_dir)
    plot_generalization_gap(data, out_dir)
    print("All 4 figures generated successfully!")


if __name__ == "__main__":
    main()
