"""Chạy thí nghiệm như mục 5 của paper: tỉ lệ instance giải tối ưu và thời gian chạy theo bản đồ, số agent.

Ví dụ:
  # toàn bộ 1.350 instance, cả hai biến thể, timeout 5 phút như paper (rất lâu!)
  .venv/bin/python run_experiments.py --time-limit 300

  # chạy thử nhanh: bản đồ 20x20, 20 và 30 agent, 5 instance đầu, timeout 60 giây
  .venv/bin/python run_experiments.py --maps 20x20 --agents 20 30 --instances 5 --time-limit 60

Mỗi lần chạy xong một instance, một dòng được ghi thêm vào file CSV (mặc định results/results.csv);
chạy lại cùng lệnh sẽ bỏ qua các dòng đã có, nên có thể dừng và chạy tiếp.
"""
import argparse
import csv
import os
import sys
import time

from bcp_mapf import solve
from generate_dataset import AGENT_COUNTS, MAPS, N_INSTANCES, SCEN
from mapf_io import load_map, load_scen, sum_of_costs, validate_solution

FIELDS = ["map", "agents", "instance", "solver", "status", "soc", "lb", "root_lb", "time",
          "nodes", "columns", "vertex_cuts", "edge_cuts", "corridor_cuts", "rectangle_cuts",
          "length_branches", "vertex_branches", "valid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=list(AGENT_COUNTS))
    ap.add_argument("--agents", nargs="+", type=int, help="mặc định: các số agent của paper")
    ap.add_argument("--instances", type=int, default=N_INSTANCES, help="số instance đầu tiên mỗi cấu hình")
    ap.add_argument("--solvers", nargs="+", default=["BCP", "BCP-B"], choices=["BCP", "BCP-B"])
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "results.csv"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/N: chỉ chạy các instance có thứ tự %% N == i (để chạy song song)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["map"], int(r["agents"]), int(r["instance"]), r["solver"]) for r in csv.DictReader(f)}
    new_file = not os.path.exists(args.out)
    out = open(args.out, "a", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
    if new_file:
        w.writeheader()

    shard, n_shards = map(int, args.shard.split("/"))
    task_no = -1
    grids = {}
    for name in args.maps:
        grid = grids.setdefault(name, load_map(os.path.join(MAPS, name + ".map")))
        for k in args.agents or AGENT_COUNTS[name]:
            for i in range(args.instances):
                task_no += 1
                if task_no % n_shards != shard:
                    continue
                scen = os.path.join(SCEN, name, f"{name}-{k}agents-{i}.scen")
                if not os.path.exists(scen):
                    print(f"bỏ qua (không có file): {scen}", file=sys.stderr)
                    continue
                _, agents = load_scen(scen)
                for solver in args.solvers:
                    if (name, k, i, solver) in done:
                        continue
                    t = time.perf_counter()
                    r = solve(grid, agents, solver, args.time_limit, args.verbose)
                    valid = ""
                    if r["paths"] is not None:
                        errs = validate_solution(grid, agents, r["paths"])
                        valid = "yes" if not errs and sum_of_costs(r["paths"]) == r["soc"] else "NO"
                    row = {"map": name, "agents": k, "instance": i, "solver": solver, **r,
                           "time": round(r["time"], 3), "valid": valid}
                    if isinstance(row.get("root_lb"), float):
                        row["root_lb"] = round(row["root_lb"], 4)
                    w.writerow(row)
                    out.flush()
                    print(f"{name:12s} k={k:3d} #{i:02d} {solver:5s} {r['status']:8s} "
                          f"SOC={r['soc']} LB={r['lb']} t={time.perf_counter() - t:7.2f}s nodes={r.get('nodes', 0)}",
                          flush=True)
    out.close()


if __name__ == "__main__":
    main()
