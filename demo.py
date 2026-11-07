#!/usr/bin/env python3
"""
demo.py – Chương trình minh hoạ và so sánh ba thuật toán định tuyến hai robot:
1. BFS không gian chung (BFS Joint) – tối ưu makespan.
2. A* Độc lập + Lập lịch BFS trên tiến trình đường đi cố định (A* Independent + Finite BFS Coordination).
3. Joint A* với Heuristic Max Manhattan (Joint A*) – tối ưu makespan.

Bao gồm:
- Kịch bản 1: Kho hàng 10×10 với các dãy kệ hàng (Shelves). Cả 3 thuật toán đều thành công.
- Kịch bản 2: Hành lang hẹp có vịnh tránh xe (Passing Bay). Minh hoạ giới hạn của A* độc lập
  (thất bại vì đường đi cố định không thể tránh nhau), trong khi Joint A* và BFS thành công
  nhờ điều hướng tạm thời vào vịnh tránh.
- Tùy chọn dòng lệnh: cờ `--steps` để hiển thị trực quan từng bước di chuyển trên lưới.

Sử dụng:
    python3 demo.py
    python3 demo.py --steps
    python3 demo.py --scenario warehouse --steps
    python3 demo.py --scenario bay --steps

Tính tái lập (Reproducibility):
- Toàn bộ dự án KHÔNG sử dụng hàm ngẫu nhiên (random) nên không cần seed.
- Thứ tự duyệt hành động cố định; heap của A* phá hoà bằng bộ đếm tăng dần (counter),
  do đó Makespan, số bước di chuyển và số trạng thái mở luôn giống nhau ở mọi lần chạy.
- Chỉ cột thời gian chạy (ms, đo bằng time.perf_counter) dao động theo máy và tải hệ thống.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from common import (
    Metrics,
    Position,
    format_grid,
    print_path_steps,
    validate_joint_path,
)
from bfs import bfs_joint
from astar import astar_independent
from joint_astar import joint_astar


def create_warehouse_grid() -> List[List[int]]:
    """Tạo lưới kho hàng 10×10 với hai khối kệ hàng song song.

    Bố trí kệ hàng: cột 2-3 và cột 6-7, hàng 2-7.
       0 1 2 3 4 5 6 7 8 9
    0  . . . . . . . . . .
    1  . . . . . . . . . .
    2  . . # # . . # # . .
    3  . . # # . . # # . .
    4  . . # # . . # # . .
    5  . . # # . . # # . .
    6  . . # # . . # # . .
    7  . . # # . . # # . .
    8  . . . . . . . . . .
    9  . . . . . . . . . .
    """
    grid = [[0] * 10 for _ in range(10)]
    for y in range(2, 8):
        grid[y][2] = 1
        grid[y][3] = 1
        grid[y][6] = 1
        grid[y][7] = 1
    return grid


def create_passing_bay_grid() -> List[List[int]]:
    """Tạo lưới hành lang hẹp 3×5 có vịnh tránh xe (passing bay) ở ô (2, 0) và (2, 2).

       0 1 2 3 4
    0  # # . # #
    1  . . . . .
    2  # # . # #
    """
    return [
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
    ]


def print_separator(char: str = "─", length: int = 76) -> None:
    print(char * length)


def print_metrics_table(
    results: List[Dict[str, Any]],
    title: str = "BẢNG SO SÁNH HIỆU SUẤT",
) -> None:
    """In bảng so sánh các chỉ số đo lường hiệu suất."""
    print_separator("═")
    print(f"  {title}")
    print_separator("═")

    headers = [
        "Thuật toán",
        "Makespan",
        "Di chuyển A",
        "Di chuyển B",
        "Trạng thái mở",
        "Thời gian (ms)",
        "Trạng thái",
    ]
    widths = [20, 10, 12, 12, 15, 16, 12]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print_separator("─", len(header_line))

    for r in results:
        res = r["result"]
        name = r["name"]
        if res is None:
            fail = r.get("failure") or {}
            expanded = (
                str(fail["expanded_states"]) if "expanded_states" in fail else "N/A"
            )
            runtime = (
                f"{fail['runtime'] * 1000:.3f}" if "runtime" in fail else "N/A"
            )
            row = [
                name.ljust(widths[0]),
                "N/A".center(widths[1]),
                "N/A".center(widths[2]),
                "N/A".center(widths[3]),
                expanded.center(widths[4]),
                runtime.center(widths[5]),
                "Thất bại".center(widths[6]),
            ]
        else:
            m: Metrics = res["metrics"]
            row = [
                m.algorithm.ljust(widths[0]),
                str(m.makespan).center(widths[1]),
                str(m.path_length_a).center(widths[2]),
                str(m.path_length_b).center(widths[3]),
                str(m.expanded_states).center(widths[4]),
                f"{m.runtime_seconds * 1000:.3f}".center(widths[5]),
                "Thành công".center(widths[6]),
            ]
        print(" | ".join(row))

    print_separator("─", len(header_line))


def print_stage_metrics(result: Dict[str, Any]) -> None:
    """In chi tiết các giai đoạn của thuật toán A* Độc lập."""
    if "stage_metrics" not in result:
        return
    sm = result["stage_metrics"]
    print("\n  📊 Chi tiết từng giai đoạn của A* Độc lập:")
    print(
        f"    • Giai đoạn 1A (A* đơn lẻ cho Robot A): "
        f"{sm['search_a']['expanded']} trạng thái, "
        f"{sm['search_a']['runtime'] * 1000:.3f} ms, "
        f"độ dài đường đi = {sm['search_a']['path_length']} bước"
    )
    print(
        f"    • Giai đoạn 1B (A* đơn lẻ cho Robot B): "
        f"{sm['search_b']['expanded']} trạng thái, "
        f"{sm['search_b']['runtime'] * 1000:.3f} ms, "
        f"độ dài đường đi = {sm['search_b']['path_length']} bước"
    )
    print(
        f"    • Giai đoạn 2  (Phối hợp trên chỉ số cố định i, j): "
        f"{sm['coordination']['expanded']} trạng thái, "
        f"{sm['coordination']['runtime'] * 1000:.3f} ms"
    )


def run_scenario_warehouse(show_steps: bool = False) -> None:
    """Kịch bản 1: Kho hàng 10x10 tiêu chuẩn.

    Chọn điểm xuất phát và đích sao cho cả 3 thuật toán đều thành công,
    thể hiện rõ sự tối ưu của BFS / Joint A* so với A* độc lập.
    """
    grid = create_warehouse_grid()
    start_a: Position = (0, 0)
    goal_a: Position = (5, 9)
    start_b: Position = (4, 0)
    goal_b: Position = (0, 9)

    print("\n" + "█" * 76)
    print("  KỊCH BẢN 1: KHO HÀNG 10×10 VỚI KỆ HÀNG (SHELVES)")
    print("  (Cả ba thuật toán đều tìm được đường đi thành công)")
    print("█" * 76)

    print("\n📦 Bản đồ kho hàng ban đầu:")
    print(format_grid(
        grid,
        start_a=start_a,
        goal_a=goal_a,
        start_b=start_b,
        goal_b=goal_b,
    ))
    print(f"\n  • Robot A (a -> α): xuất phát {start_a} -> đích {goal_a}")
    print(f"  • Robot B (b -> β): xuất phát {start_b} -> đích {goal_b}")

    results: List[Dict[str, Any]] = []

    # 1. BFS Joint
    print_separator()
    print("▶ 1. Chạy BFS Không gian chung (BFS Joint)...")
    res_bfs = bfs_joint(grid, start_a, goal_a, start_b, goal_b)
    results.append({"name": "BFS (Joint)", "result": res_bfs})
    if res_bfs:
        validate_joint_path(
            grid, res_bfs["joint_path"], start_a, goal_a, start_b, goal_b
        )
        print(f"   ✔ Thành công! Makespan = {res_bfs['metrics'].makespan} (Tối ưu toàn cục).")
        print("   ✔ Kế hoạch đã qua kiểm tra hợp lệ (không xung đột đỉnh/cạnh).")
    else:
        print("   ✖ Không tìm thấy lời giải.")

    # 2. A* Độc lập
    print_separator()
    print("▶ 2. Chạy A* Độc lập + Lập lịch phối hợp...")
    res_ind = astar_independent(grid, start_a, goal_a, start_b, goal_b)
    results.append({"name": "A* Độc lập", "result": res_ind})
    if res_ind:
        validate_joint_path(
            grid, res_ind["joint_path"], start_a, goal_a, start_b, goal_b
        )
        print(f"   ✔ Thành công! Makespan = {res_ind['metrics'].makespan}.")
        print("   ✔ Kế hoạch đã qua kiểm tra hợp lệ (không xung đột đỉnh/cạnh).")
        print_stage_metrics(res_ind)
    else:
        print("   ✖ Không thể phối hợp các đường đi cố định.")

    # 3. Joint A*
    print_separator()
    print("▶ 3. Chạy Joint A* (Heuristic Max Manhattan)...")
    res_joint = joint_astar(grid, start_a, goal_a, start_b, goal_b)
    results.append({"name": "Joint A*", "result": res_joint})
    if res_joint:
        validate_joint_path(
            grid, res_joint["joint_path"], start_a, goal_a, start_b, goal_b
        )
        print(f"   ✔ Thành công! Makespan = {res_joint['metrics'].makespan} (Tối ưu toàn cục).")
        print("   ✔ Kế hoạch đã qua kiểm tra hợp lệ (không xung đột đỉnh/cạnh).")
    else:
        print("   ✖ Không tìm thấy lời giải.")

    # Bảng so sánh
    print()
    print_metrics_table(results, "BẢNG SO SÁNH KỊCH BẢN 1 (KHO HÀNG 10×10)")

    print("\n💡 Nhận xét Kịch bản 1:")
    print("  • BFS Joint và Joint A* đều tìm ra makespan tối ưu = 14.")
    print("  • Joint A* duyệt ít hơn đáng kể số trạng thái so với BFS nhờ heuristic admissible.")
    print("  • A* Độc lập chạy nhanh nhất ở khâu đơn lẻ nhưng có makespan lớn hơn (18 > 14)")
    print("    do phải chèn nhiều bước đợi trên đường đi đơn lẻ cố định, không tìm đường vòng tối ưu.")

    if show_steps and res_joint:
        print("\n" + "=" * 76)
        print("  CHI TIẾT TỪNG BƯỚC DI CHUYỂN TRÊN LƯỚI (JOINT A* - TỐI ƯU)")
        print("=" * 76)
        print_path_steps(
            grid,
            res_joint["joint_path"],
            start_a=start_a,
            goal_a=goal_a,
            start_b=start_b,
            goal_b=goal_b,
        )


def run_scenario_passing_bay(show_steps: bool = False) -> None:
    """Kịch bản 2: Hành lang hẹp có vịnh tránh xe (Passing Bay).

    Hai robot đi ngược chiều trên hành lang 1 làn.
    Đường đi ngắn nhất độc lập không có chỗ tránh nếu chỉ chèn bước đợi.
    Joint A* và BFS giải được nhờ điều hướng một robot tạm thời rẽ vào vịnh tránh.
    """
    grid = create_passing_bay_grid()
    start_a: Position = (0, 1)
    goal_a: Position = (4, 1)
    start_b: Position = (4, 1)
    goal_b: Position = (0, 1)

    print("\n" + "█" * 76)
    print("  KỊCH BẢN 2: HÀNH LANG HẸP CÓ VỊNH TRÁNH XE (PASSING BAY)")
    print("  (A* Độc lập THẤT BẠI; BFS và Joint A* THÀNH CÔNG nhờ né sang vịnh)")
    print("█" * 76)

    print("\n📦 Bản đồ hành lang có vịnh tránh (ô (2,0) và (2,2)):")
    print(format_grid(
        grid,
        start_a=start_a,
        goal_a=goal_a,
        start_b=start_b,
        goal_b=goal_b,
    ))
    print(f"\n  • Robot A: xuất phát {start_a} -> đích {goal_a} (đi từ trái sang phải)")
    print(f"  • Robot B: xuất phát {start_b} -> đích {goal_b} (đi từ phải sang trái)")
    print("  • Hành lang chỉ có 1 làn tại hàng y=1; muốn tránh nhau phải rẽ vào vịnh y=0 hoặc y=2.")

    results: List[Dict[str, Any]] = []

    # 1. BFS Joint
    print_separator()
    print("▶ 1. Chạy BFS Không gian chung (BFS Joint)...")
    res_bfs = bfs_joint(grid, start_a, goal_a, start_b, goal_b)
    results.append({"name": "BFS (Joint)", "result": res_bfs})
    if res_bfs:
        validate_joint_path(
            grid, res_bfs["joint_path"], start_a, goal_a, start_b, goal_b
        )
        print(f"   ✔ Thành công! Makespan = {res_bfs['metrics'].makespan}.")
    else:
        print("   ✖ Không tìm thấy lời giải.")

    # 2. A* Độc lập
    print_separator()
    print("▶ 2. Chạy A* Độc lập + Lập lịch phối hợp...")
    fail_ind: Dict[str, Any] = {}
    res_ind = astar_independent(
        grid, start_a, goal_a, start_b, goal_b, failure_stats=fail_ind
    )
    results.append({"name": "A* Độc lập", "result": res_ind, "failure": fail_ind})
    if res_ind:
        validate_joint_path(
            grid, res_ind["joint_path"], start_a, goal_a, start_b, goal_b
        )
        print(f"   ✔ Thành công! Makespan = {res_ind['metrics'].makespan}.")
    else:
        print("   ✖ Thất bại (Trả về None như thiết kế):")
        print("     Đường đi ngắn nhất của cả hai robot đều là đường thẳng y=1.")
        print("     Thuật toán lập lịch chỉ chèn bước đợi trên đường đi cố định,")
        print("     không thể tự tạo đường vòng rẽ vào vịnh tránh xe.")
        if fail_ind.get("failed_stage") == "coordination":
            print_stage_metrics(fail_ind)

    # 3. Joint A*
    print_separator()
    print("▶ 3. Chạy Joint A* (Heuristic Max Manhattan)...")
    res_joint = joint_astar(grid, start_a, goal_a, start_b, goal_b)
    results.append({"name": "Joint A*", "result": res_joint})
    if res_joint:
        validate_joint_path(
            grid, res_joint["joint_path"], start_a, goal_a, start_b, goal_b
        )
        print(f"   ✔ Thành công! Makespan = {res_joint['metrics'].makespan}.")
    else:
        print("   ✖ Không tìm thấy lời giải.")

    # Bảng so sánh
    print()
    print_metrics_table(results, "BẢNG SO SÁNH KỊCH BẢN 2 (PASSING BAY)")

    print("\n💡 Nhận xét Kịch bản 2:")
    print("  • Minh hoạ rõ rệt giới hạn bản chất của phương pháp đường đi cố định (Fixed-Path Coordination):")
    print("    Nếu đường đi ngắn nhất ban đầu đối đầu trực diện trong không gian hẹp, việc chỉ chèn")
    print("    bước đợi (wait) không thể giải quyết bế tắc (deadlock).")
    print("  • Tìm kiếm trên không gian trạng thái chung (BFS Joint / Joint A*) hoàn chỉnh (complete)")
    print("    và tìm được phương án cho một robot tạm lùi/rẽ vào vịnh nhường đường rồi đi tiếp.")

    if show_steps and res_joint:
        print("\n" + "=" * 76)
        print("  CHI TIẾT TỪNG BƯỚC NÉ TRÁNH TRÊN LƯỚI (JOINT A*):")
        print("=" * 76)
        print_path_steps(
            grid,
            res_joint["joint_path"],
            start_a=start_a,
            goal_a=goal_a,
            start_b=start_b,
            goal_b=goal_b,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minh hoạ và so sánh ba thuật toán định tuyến hai robot trong kho hàng."
    )
    parser.add_argument(
        "--steps",
        action="store_true",
        help="Hiển thị trực quan từng bước di chuyển trên lưới.",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", "warehouse", "bay"],
        default="all",
        help="Chọn kịch bản thực thi: warehouse (kho hàng 10x10), bay (vịnh tránh), hoặc all (cả hai).",
    )

    args = parser.parse_args()

    print("\n" + "=" * 76)
    print("  CHƯƠNG TRÌNH SO SÁNH CÁC THUẬT TOÁN ĐỊNH TUYẾN HAI ROBOT (MAPF)")
    print("  Thực hiện bằng Python Standard Library")
    print("=" * 76)

    if args.scenario in ("all", "warehouse"):
        run_scenario_warehouse(show_steps=args.steps)

    if args.scenario in ("all", "bay"):
        run_scenario_passing_bay(show_steps=args.steps)

    print("\n" + "=" * 76)
    print("  DEMO HOÀN TẤT THÀNH CÔNG")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
