"""
bfs.py – BFS trên không gian trạng thái chung cho hai robot.

Thuật toán BFS tìm kiếm theo chiều rộng trên trạng thái chung
((xA, yA), (xB, yB)), đảm bảo tối ưu makespan (số bước đồng bộ
ít nhất) vì mỗi cạnh có chi phí 1.

Robot được phép rời đích của mình và quay lại; chỉ trạng thái chung
cuối cùng phải khớp với đích.
"""

from __future__ import annotations

import time
from collections import deque
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from common import (
    DIRECTIONS,
    Grid,
    JointState,
    Metrics,
    Position,
    compute_metrics,
    has_edge_swap_conflict,
    has_vertex_conflict,
    reconstruct_path,
    validate_inputs,
)


def bfs_joint(
    grid: List[List[int]],
    start_a: Position,
    goal_a: Position,
    start_b: Position,
    goal_b: Position,
) -> Optional[Dict[str, Any]]:
    """BFS tìm đường đi tối ưu makespan cho hai robot.

    Tìm kiếm trên không gian trạng thái chung với 25 hành động (5×5)
    mỗi bước đồng bộ. Chi phí mỗi bước = 1.

    Args:
        grid: Lưới 2D, grid[y][x] ∈ {0, 1}.
        start_a: Vị trí bắt đầu robot A (x, y).
        goal_a: Vị trí đích robot A (x, y).
        start_b: Vị trí bắt đầu robot B (x, y).
        goal_b: Vị trí đích robot B (x, y).

    Returns:
        Dict chứa:
            'joint_path': danh sách trạng thái chung,
            'path_a': đường đi robot A,
            'path_b': đường đi robot B,
            'metrics': đối tượng Metrics.
        Hoặc None nếu không có lời giải.

    Raises:
        ValueError: nếu đầu vào không hợp lệ.
    """
    # ── Kiểm tra đầu vào ──
    g = validate_inputs(grid, start_a, goal_a, start_b, goal_b)

    t_start = time.perf_counter()

    # ── Trường hợp đặc biệt: cả hai đã ở đích ──
    if start_a == goal_a and start_b == goal_b:
        joint_path: List[JointState] = [(start_a, start_b)]
        runtime = time.perf_counter() - t_start
        return {
            "joint_path": joint_path,
            "path_a": [start_a],
            "path_b": [start_b],
            "metrics": compute_metrics(
                algorithm_name="BFS (Joint)",
                expanded_states=0,
                runtime=runtime,
                joint_path=joint_path,
            ),
        }

    # ── BFS ──
    initial: JointState = (start_a, start_b)
    goal_state: JointState = (goal_a, goal_b)

    queue: deque[JointState] = deque([initial])
    visited: set[JointState] = {initial}
    parents: Dict[JointState, JointState] = {}
    expanded = 0

    while queue:
        current = queue.popleft()
        expanded += 1

        if current == goal_state:
            # Dựng lại đường đi
            joint_path = reconstruct_path(parents, current)
            runtime = time.perf_counter() - t_start
            path_a = [s[0] for s in joint_path]
            path_b = [s[1] for s in joint_path]
            return {
                "joint_path": joint_path,
                "path_a": path_a,
                "path_b": path_b,
                "metrics": compute_metrics(
                    algorithm_name="BFS (Joint)",
                    expanded_states=expanded,
                    runtime=runtime,
                    joint_path=joint_path,
                ),
            }

        pos_a, pos_b = current

        # Tạo tất cả 25 hành động chung
        for (da_x, da_y), (db_x, db_y) in product(DIRECTIONS, DIRECTIONS):
            new_a: Position = (pos_a[0] + da_x, pos_a[1] + da_y)
            new_b: Position = (pos_b[0] + db_x, pos_b[1] + db_y)

            # Kiểm tra biên và chướng ngại vật
            if not g.is_free(new_a[0], new_a[1]):
                continue
            if not g.is_free(new_b[0], new_b[1]):
                continue

            # Kiểm tra xung đột đỉnh (cùng ô)
            if has_vertex_conflict(new_a, new_b):
                continue

            # Kiểm tra xung đột hoán đổi cạnh
            if has_edge_swap_conflict(pos_a, new_a, pos_b, new_b):
                continue

            successor: JointState = (new_a, new_b)
            if successor not in visited:
                visited.add(successor)
                parents[successor] = current
                queue.append(successor)

    # Không tìm thấy lời giải
    runtime = time.perf_counter() - t_start
    return None
