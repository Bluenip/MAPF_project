#!/usr/bin/env python3
"""
benchmark.py – Tái lập bảng kết quả trên slide 18 và 23.

Quy trình đo (giống mô tả ở slide 16 "Experimental Setup"):
- Mỗi thuật toán chạy 1 lần khởi động (warm-up, không tính), sau đó 30 lần đo.
- Runtime lấy từ chính thuật toán (time.perf_counter), KHÔNG tính bước validate_inputs.
- Mọi kế hoạch tìm được đều được kiểm tra bằng validate_joint_path (ngoài phần đo thời gian).
- Báo cáo trung vị (median): sắp xếp 30 giá trị, lấy trung bình giá trị thứ 15 và 16.

Tính tái lập:
- Không dùng hàm ngẫu nhiên (random) nên không cần seed.
- Makespan, số bước di chuyển và số trạng thái mở là TẤT ĐỊNH, luôn giống nhau mọi lần chạy.
- Chỉ runtime (ms) dao động theo phần cứng / tải máy.

Sử dụng:
    python3 benchmark.py            # 30 lần đo (mặc định)
    python3 benchmark.py --runs 50
"""

from __future__ import annotations

import argparse
import platform
import re
import statistics
from typing import Any, Callable, Dict, List

from astar import astar_independent
from bfs import bfs_joint
from common import validate_joint_path
from demo import create_passing_bay_grid, create_warehouse_grid
from joint_astar import joint_astar

SCENARIOS = [
    ("Scenario 1: 10x10 warehouse", create_warehouse_grid, (0, 0), (5, 9), (4, 0), (0, 9)),
    ("Scenario 2: 3x5 passing bay", create_passing_bay_grid, (0, 1), (4, 1), (4, 1), (0, 1)),
]


def _run_once(fn: Callable[..., Any], name: str, grid, sa, ga, sb, gb) -> Dict[str, Any]:
    """Chạy thuật toán một lần, trả về các chỉ số (kể cả khi thất bại)."""
    if name == "Independent A* + scheduling":
        fail: Dict[str, Any] = {}
        res = fn(grid, sa, ga, sb, gb, failure_stats=fail)
        if res is None:
            return {"ok": False, "T": None, "moves": None,
                    "expanded": fail.get("expanded_states"), "runtime": fail.get("runtime")}
    else:
        res = fn(grid, sa, ga, sb, gb)
        if res is None:
            return {"ok": False, "T": None, "moves": None, "expanded": None, "runtime": None}
    # Kiểm tra kế hoạch (đúng đích, mỗi bước đi sang ô kề hoặc đợi, không ra ngoài biên,
    # không vào kệ hàng, không xung đột đỉnh/hoán đổi cạnh). Chạy SAU khi thuật toán đã đo
    # thời gian xong, nên không ảnh hưởng runtime. Kế hoạch sai sẽ ném ValueError.
    validate_joint_path(grid, res["joint_path"], sa, ga, sb, gb)
    m = res["metrics"]
    return {"ok": True, "T": m.makespan, "moves": (m.path_length_a, m.path_length_b),
            "expanded": m.expanded_states, "runtime": m.runtime_seconds}


def _cpu_name() -> str:
    """Tên CPU (chỉ để ghi lại môi trường đo, không ảnh hưởng kết quả)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            m = re.search(r"model name\s*:\s*(.+)", f.read())
            if m:
                return m.group(1).strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=30, help="Số lần đo cho mỗi thuật toán (mặc định 30)")
    args = ap.parse_args()

    algos = [
        ("Joint BFS", bfs_joint),
        ("Independent A* + scheduling", astar_independent),
        ("Joint A*", joint_astar),
    ]

    print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()} | {_cpu_name()}")
    print(f"1 warm-up + {args.runs} timed runs per algorithm; reported runtime = median\n")

    for title, make_grid, sa, ga, sb, gb in SCENARIOS:
        grid = make_grid()
        print(title)
        print(f"{'Algorithm':<30}{'Outcome':<20}{'T':>4}{'Moves A/B':>12}{'Expanded':>10}{'Median ms':>12}")
        for name, fn in algos:
            _run_once(fn, name, grid, sa, ga, sb, gb)  # warm-up
            runs: List[Dict[str, Any]] = [_run_once(fn, name, grid, sa, ga, sb, gb) for _ in range(args.runs)]
            # Các chỉ số tất định phải giống hệt nhau ở mọi lần chạy.
            first = runs[0]
            assert all((r["ok"], r["T"], r["moves"], r["expanded"]) ==
                       (first["ok"], first["T"], first["moves"], first["expanded"]) for r in runs)
            med_ms = statistics.median(r["runtime"] for r in runs) * 1000
            outcome = "Success" if first["ok"] else "No valid schedule"
            t = str(first["T"]) if first["ok"] else "-"
            mv = f"{first['moves'][0]} / {first['moves'][1]}" if first["ok"] else "-"
            print(f"{name:<30}{outcome:<20}{t:>4}{mv:>12}{first['expanded']:>10,}{med_ms:>12.3f}")
        print()


if __name__ == "__main__":
    main()
