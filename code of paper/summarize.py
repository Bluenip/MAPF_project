"""Gộp kết quả các shard và tạo bảng + biểu đồ có dạng giống Hình 3 và Hình 4 của paper
(trên bộ dữ liệu mô phỏng, nên số liệu không so sánh trực tiếp được với paper).

Chạy:  .venv/bin/python summarize.py
Đầu ra: results/results_all.csv, results/summary.md, results/fig3_success_rate.png, results/fig4_runtime.png
"""
import csv
import glob
import os
from collections import defaultdict

from generate_dataset import AGENT_COUNTS, N_INSTANCES

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
SOLVERS = ["BCP", "BCP-B"]


def load():
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(RES, "results_shard*.csv"))) + glob.glob(os.path.join(RES, "results.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                key = (r["map"], int(r["agents"]), int(r["instance"]), r["solver"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    return rows


def main():
    rows = load()
    if not rows:
        print("Chưa có kết quả.")
        return
    with open(os.path.join(RES, "results_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = {(r["map"], int(r["agents"]), int(r["instance"]), r["solver"]): r for r in rows}
    total = sum(len(v) for v in AGENT_COUNTS.values()) * N_INSTANCES
    bad = [r for r in rows if r["valid"] == "NO"]
    lines = ["# Kết quả chạy BCP/BCP-B trên bộ dữ liệu mô phỏng theo Lam et al. (IJCAI 2019)", ""]
    for s in SOLVERS:
        done = [r for r in rows if r["solver"] == s]
        opt = [r for r in done if r["status"] == "optimal"]
        lines.append(f"- **{s}**: đã chạy {len(done)}/{total}, giải tối ưu {len(opt)}"
                     + (f" ({100 * len(opt) / len(done):.1f}% số đã chạy)" if done else ""))
    lines.append(f"- Lời giải không hợp lệ (validator): {len(bad)}")
    # kiểm tra chéo: khi cả hai cùng tối ưu thì SOC phải bằng nhau
    mism = [k for k in by if k[3] == "BCP" and (k[:3] + ("BCP-B",)) in by
            and by[k]["status"] == by[k[:3] + ("BCP-B",)]["status"] == "optimal"
            and by[k]["soc"] != by[k[:3] + ("BCP-B",)]["soc"]]
    lines.append(f"- Instance cả hai cùng tối ưu nhưng SOC khác nhau: {len(mism)}")
    lines += ["", "## Tỉ lệ giải tối ưu (Hình 3) và thời gian trung bình trên các instance cả hai cùng giải (Hình 4)", "",
              "| Bản đồ | Agents | Đã chạy | BCP | BCP-B | Thời gian BCP (s) | Thời gian BCP-B (s) |",
              "|---|---|---|---|---|---|---|"]
    fig3 = defaultdict(dict)
    fig4 = defaultdict(dict)
    for name, ks in AGENT_COUNTS.items():
        for k in ks:
            cnt = {s: 0 for s in SOLVERS}
            ran = 0
            both = []
            for i in range(N_INSTANCES):
                rs = {s: by.get((name, k, i, s)) for s in SOLVERS}
                if all(rs.values()):
                    ran += 1
                    for s in SOLVERS:
                        cnt[s] += rs[s]["status"] == "optimal"
                    if all(rs[s]["status"] == "optimal" for s in SOLVERS):
                        both.append({s: float(rs[s]["time"]) for s in SOLVERS})
            if not ran:
                continue
            avg = {s: (sum(b[s] for b in both) / len(both)) if both else None for s in SOLVERS}
            for s in SOLVERS:
                fig3[name][k] = fig3[name].get(k, {}) | {s: 100 * cnt[s] / ran}
                fig4[name][k] = fig4[name].get(k, {}) | {s: avg[s]}
            fmt = lambda x: "—" if x is None else f"{x:.2f}"
            lines.append(f"| {name} | {k} | {ran}/50 | {cnt['BCP']} ({100 * cnt['BCP'] / ran:.0f}%) | "
                         f"{cnt['BCP-B']} ({100 * cnt['BCP-B'] / ran:.0f}%) | {fmt(avg['BCP'])} | {fmt(avg['BCP-B'])} |")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for fig_data, fname, ylabel in ((fig3, "fig3_success_rate.png", "Percentage solved (%)"),
                                    (fig4, "fig4_runtime.png", "Time (s)")):
        maps = [m for m in AGENT_COUNTS if m in fig_data]
        fig, axes = plt.subplots(1, len(maps), figsize=(4 * len(maps), 3.2), squeeze=False)
        for ax, m in zip(axes[0], maps):
            ks = sorted(fig_data[m])
            for s, style in zip(SOLVERS, ("o-", "s--")):
                ys = [fig_data[m][k].get(s) for k in ks]
                ax.plot([k for k, y in zip(ks, ys) if y is not None], [y for y in ys if y is not None], style, label=s)
            ax.set_title(m)
            ax.set_xlabel("Agents")
            ax.set_ylabel(ylabel)
            if "success" in fname:
                ax.set_ylim(-5, 105)
            ax.grid(alpha=.3)
        axes[0][0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RES, fname), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
