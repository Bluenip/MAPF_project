"""
joint_astar.py – Joint A* cho hai robot trên không gian trạng thái chung.

Thuật toán mở rộng A* truyền thống bằng cách coi hai robot như một
tác nhân tổ hợp. Mỗi trạng thái = ((xA, yA), (xB, yB)), mỗi
chuyển tiếp = hành động chung (aA, aB).

Heuristic: h(s) = max(Manhattan(posA, goalA), Manhattan(posB, goalB))
- Admissible (chấp nhận được) và consistent (nhất quán).
- Tối ưu makespan.

Robot được phép rời đích; chỉ trạng thái chung cuối phải = goal.
"""

from __future__ import annotations

import heapq
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from common import (
    DIRECTIONS,
    JointState,
    Position,
    compute_metrics,
    has_edge_swap_conflict,
    has_vertex_conflict,
    makespan_heuristic,
    reconstruct_path,
    validate_inputs,
)


def joint_astar(
    grid: List[List[int]],
    start_a: Position,
    goal_a: Position,
    start_b: Position,
    goal_b: Position,
) -> Optional[Dict[str, Any]]:
    """Joint A* tìm đường tối ưu makespan cho hai robot.

    Tìm kiếm trên không gian trạng thái chung ((xA,yA),(xB,yB))
    với heuristic makespan = max(Manhattan_A, Manhattan_B).

    Args:
        grid: Lưới 2D, grid[y][x] ∈ {0, 1}.
        start_a: Vị trí bắt đầu robot A (x, y).
        goal_a: Vị trí đích robot A (x, y).
        start_b: Vị trí bắt đầu robot B (x, y).
        goal_b: Vị trí đích robot B (x, y).

    Returns:
        Dict chứa 'joint_path', 'path_a', 'path_b', 'metrics'.
        Hoặc None nếu không có lời giải.

    Raises:
        ValueError: nếu đầu vào không hợp lệ.
    """
    # ── Kiểm tra đầu vào ──
    g = validate_inputs(grid, start_a, goal_a, start_b, goal_b)

    t_start = time.perf_counter()

    # ── Trường hợp đặc biệt ──
    if start_a == goal_a and start_b == goal_b:
        joint_path: List[JointState] = [(start_a, start_b)]
        runtime = time.perf_counter() - t_start
        return {
            "joint_path": joint_path,
            "path_a": [start_a],
            "path_b": [start_b],
            "metrics": compute_metrics(
                algorithm_name="Joint A*",
                expanded_states=0,
                runtime=runtime,
                joint_path=joint_path,
            ),
        }

    # ── Khởi tạo ──
    initial: JointState = (start_a, start_b)
    goal_state: JointState = (goal_a, goal_b)

    g_score: Dict[JointState, int] = {initial: 0}
    h_initial = makespan_heuristic(initial, goal_a, goal_b)

    # Heap: (f_score, counter, state)
    # counter để tránh so sánh trực tiếp giữa các state khi f bằng nhau
    counter = 0
    open_heap: List[Tuple[int, int, JointState]] = [
        (h_initial, counter, initial)
    ]

    parents: Dict[JointState, JointState] = {}
    closed: set[JointState] = set()
    expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal_state:
            joint_path = reconstruct_path(parents, current)
            runtime = time.perf_counter() - t_start
            path_a = [s[0] for s in joint_path]
            path_b = [s[1] for s in joint_path]
            return {
                "joint_path": joint_path,
                "path_a": path_a,
                "path_b": path_b,
                "metrics": compute_metrics(
                    algorithm_name="Joint A*",
                    expanded_states=expanded,
                    runtime=runtime,
                    joint_path=joint_path,
                ),
            }

        pos_a, pos_b = current
        current_g = g_score[current]

        # Tạo tất cả 25 hành động chung
        for (da_x, da_y), (db_x, db_y) in product(DIRECTIONS, DIRECTIONS):
            new_a: Position = (pos_a[0] + da_x, pos_a[1] + da_y)
            new_b: Position = (pos_b[0] + db_x, pos_b[1] + db_y)

            # Kiểm tra biên và chướng ngại vật
            if not g.is_free(new_a[0], new_a[1]):
                continue
            if not g.is_free(new_b[0], new_b[1]):
                continue

            # Kiểm tra xung đột đỉnh
            if has_vertex_conflict(new_a, new_b):
                continue

            # Kiểm tra xung đột hoán đổi cạnh
            if has_edge_swap_conflict(pos_a, new_a, pos_b, new_b):
                continue

            successor: JointState = (new_a, new_b)

            if successor in closed:
                continue

            tent_g = current_g + 1
            if tent_g < g_score.get(successor, float("inf")):
                g_score[successor] = tent_g
                h = makespan_heuristic(successor, goal_a, goal_b)
                f_score = tent_g + h
                parents[successor] = current
                counter += 1
                heapq.heappush(
                    open_heap, (f_score, counter, successor)
                )

    # Không tìm thấy lời giải
    return None
