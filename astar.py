"""
astar.py – A* độc lập cho từng robot, sau đó phối hợp.

Thuật toán:
1. Chạy A* đơn lẻ cho mỗi robot (heuristic Manhattan).
2. Phối hợp hai đường đi cố định bằng tìm kiếm trên không gian
   (i, j) – chỉ số tiến trình trên đường đi cố định – để chèn
   các bước đợi tối ưu.

KHÔNG sử dụng phương pháp chèn đợi vô hạn kiểu greedy từ PDF.
Sử dụng BFS trên không gian (i, j) hữu hạn.

KHÔNG có đảm bảo tối ưu toàn cục – thuật toán chỉ tối ưu trên
đường đi cố định. Nếu phối hợp thất bại, có thể vẫn tồn tại
lời giải qua tìm kiếm chung (joint search) với đường vòng.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from common import (
    DIRECTIONS,
    Grid,
    JointState,
    Metrics,
    Position,
    compute_metrics,
    count_actual_moves,
    has_edge_swap_conflict,
    has_vertex_conflict,
    manhattan_distance,
    validate_inputs,
)


# ──────────────────── A* Đơn lẻ ────────────────────────────────────────

def single_agent_astar(
    grid_obj: Grid,
    start: Position,
    goal: Position,
) -> Optional[Dict[str, Any]]:
    """A* tìm đường ngắn nhất cho một robot.

    Heuristic: khoảng cách Manhattan (admissible, consistent).
    Chi phí mỗi bước = 1. Di chuyển 4 hướng + đợi.

    Args:
        grid_obj: Đối tượng Grid.
        start: Vị trí bắt đầu (x, y).
        goal: Vị trí đích (x, y).

    Returns:
        Dict: {'path': [positions], 'expanded_states': int, 'runtime': float}
        hoặc None nếu không có đường đi.
    """
    t_start = time.perf_counter()

    if start == goal:
        return {
            "path": [start],
            "expanded_states": 0,
            "runtime": time.perf_counter() - t_start,
        }

    # (f_score, counter, position)
    counter = 0
    g_score: Dict[Position, int] = {start: 0}
    open_heap: List[Tuple[int, int, Position]] = [
        (manhattan_distance(start, goal), counter, start)
    ]
    parents: Dict[Position, Position] = {}
    closed: set[Position] = set()
    expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            # Dựng lại đường đi
            path: List[Position] = [current]
            node = current
            while node in parents:
                node = parents[node]
                path.append(node)
            path.reverse()
            return {
                "path": path,
                "expanded_states": expanded,
                "runtime": time.perf_counter() - t_start,
            }

        cx, cy = current
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if not grid_obj.is_free(nx, ny):
                continue
            neighbor: Position = (nx, ny)
            if neighbor in closed:
                continue
            tent_g = g_score[current] + 1
            if tent_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tent_g
                f_score = tent_g + manhattan_distance(neighbor, goal)
                parents[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbor))

    return None


# ───────────────── Phối hợp đường đi cố định ───────────────────────────

def coordinate_paths(
    path_a: List[Position],
    path_b: List[Position],
    stats_out: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[List[Position], List[Position], int]]:
    """Phối hợp hai đường đi cố định bằng BFS trên không gian (i, j).

    Tìm kiếm trên các trạng thái (i, j) với i = chỉ số tiến trình
    trên path_a, j = chỉ số tiến trình trên path_b. Mỗi bước,
    mỗi robot hoặc tiến 1 bước trên đường đi cố định hoặc đợi
    (giữ nguyên chỉ số).

    Chuyển tiếp hợp lệ từ (i, j):
    - (i+1, j+1): cả hai tiến     [nếu i < N và j < M]
    - (i+1, j):   A tiến, B đợi   [nếu i < N]
    - (i, j+1):   B tiến, A đợi   [nếu j < M]
    Không cho cả hai đợi đồng thời (không tiến triển).
    Khi i = N, A đợi tại đích; khi j = M, B đợi tại đích.

    Args:
        path_a: Đường đi cố định robot A (danh sách vị trí).
        path_b: Đường đi cố định robot B (danh sách vị trí).
        stats_out: (Tùy chọn) dict để ghi số trạng thái đã duyệt
            (khoá 'expanded'), kể cả khi phối hợp thất bại.

    Returns:
        (new_path_a, new_path_b, coordination_expanded) nếu thành công.
        None nếu không thể phối hợp (hai đường đi cố định này xung đột
        không thể giải quyết bằng chèn đợi – có thể cần tìm kiếm chung
        với đường vòng).
    """
    N = len(path_a) - 1  # Chỉ số cuối cùng path_a
    M = len(path_b) - 1  # Chỉ số cuối cùng path_b

    # Trạng thái: (i, j) – chỉ số hiện tại trên path_a, path_b
    start_state = (0, 0)
    goal_state = (N, M)

    if start_state == goal_state:
        return list(path_a), list(path_b), 0

    queue: deque[Tuple[int, int]] = deque([start_state])
    visited: set[Tuple[int, int]] = {start_state}
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    expanded = 0

    while queue:
        i, j = queue.popleft()
        expanded += 1

        if (i, j) == goal_state:
            # Dựng lại chuỗi (i, j)
            seq = [(i, j)]
            state = (i, j)
            while state in parents:
                state = parents[state]
                seq.append(state)
            seq.reverse()

            # Tạo đường đi thực tế
            new_a = [path_a[s[0]] for s in seq]
            new_b = [path_b[s[1]] for s in seq]
            return new_a, new_b, expanded

        # Tạo các chuyển tiếp
        successors: List[Tuple[int, int]] = []
        can_advance_a = i < N
        can_advance_b = j < M

        if can_advance_a and can_advance_b:
            successors.append((i + 1, j + 1))  # Cả hai tiến
        if can_advance_a:
            successors.append((i + 1, j))       # A tiến, B đợi
        if can_advance_b:
            successors.append((i, j + 1))       # A đợi, B tiến
        # Khi cả hai đã ở cuối, không có chuyển tiếp → sẽ thoát vòng lặp

        # Trường hợp một robot đã ở cuối, robot kia tiến
        if not can_advance_a and can_advance_b:
            # A đợi tại đích, B tiến
            pass  # Đã thêm (i, j+1) ở trên
        if can_advance_a and not can_advance_b:
            # B đợi tại đích, A tiến
            pass  # Đã thêm (i+1, j) ở trên

        for ni, nj in successors:
            # Vị trí thực tế sau chuyển tiếp
            pos_a_curr = path_a[i]
            pos_b_curr = path_b[j]
            pos_a_next = path_a[ni]
            pos_b_next = path_b[nj]

            # Kiểm tra xung đột đỉnh
            if has_vertex_conflict(pos_a_next, pos_b_next):
                continue

            # Kiểm tra xung đột hoán đổi cạnh
            if has_edge_swap_conflict(
                pos_a_curr, pos_a_next, pos_b_curr, pos_b_next
            ):
                continue

            if (ni, nj) not in visited:
                visited.add((ni, nj))
                parents[(ni, nj)] = (i, j)
                queue.append((ni, nj))

    # Không thể phối hợp
    if stats_out is not None:
        stats_out["expanded"] = expanded
    return None


# ──────────────── A* Độc lập + Phối hợp ────────────────────────────────

def astar_independent(
    grid: List[List[int]],
    start_a: Position,
    goal_a: Position,
    start_b: Position,
    goal_b: Position,
    failure_stats: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """A* độc lập cho từng robot, sau đó phối hợp đường đi cố định.

    KHÔNG đảm bảo tối ưu toàn cục. Nếu hai đường đi ngắn nhất không
    thể phối hợp (ví dụ: hành lang hẹp), thuật toán trả về None.
    Trong trường hợp đó, Joint A* hoặc BFS có thể tìm được lời giải
    bằng đường vòng.

    Args:
        grid: Lưới 2D.
        start_a, goal_a: Bắt đầu và đích robot A.
        start_b, goal_b: Bắt đầu và đích robot B.
        failure_stats: (Tùy chọn) dict nhận số liệu khi thất bại:
            'failed_stage', 'expanded_states', 'runtime',
            'stage_metrics'. Giá trị trả về vẫn là None như cũ.

    Returns:
        Dict chứa 'joint_path', 'path_a', 'path_b', 'metrics',
        'stage_metrics' (chi tiết từng giai đoạn).
        Hoặc None nếu không tìm được lời giải.

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
                algorithm_name="A* Độc lập",
                expanded_states=0,
                runtime=runtime,
                joint_path=joint_path,
            ),
            "stage_metrics": {
                "search_a": {"expanded": 0, "runtime": 0.0, "path_length": 0},
                "search_b": {"expanded": 0, "runtime": 0.0, "path_length": 0},
                "coordination": {"expanded": 0, "runtime": 0.0},
            },
        }

    # ── Giai đoạn 1: Tìm đường A* cho mỗi robot ──
    def _record_failure(stage: str, stage_metrics: Dict[str, Any]) -> None:
        if failure_stats is None:
            return
        failure_stats["failed_stage"] = stage
        failure_stats["expanded_states"] = sum(
            s["expanded"] for s in stage_metrics.values()
        )
        failure_stats["runtime"] = time.perf_counter() - t_start
        failure_stats["stage_metrics"] = stage_metrics

    result_a = single_agent_astar(g, start_a, goal_a)
    if result_a is None:
        # single_agent_astar không trả số trạng thái khi thất bại.
        _record_failure("search_a", {})
        return None

    result_b = single_agent_astar(g, start_b, goal_b)
    if result_b is None:
        _record_failure("search_b", {
            "search_a": {
                "expanded": result_a["expanded_states"],
                "runtime": result_a["runtime"],
                "path_length": len(result_a["path"]) - 1,
            },
        })
        return None

    raw_path_a: List[Position] = result_a["path"]
    raw_path_b: List[Position] = result_b["path"]

    # ── Giai đoạn 2: Phối hợp ──
    coord_stats: Dict[str, Any] = {}
    t_coord_start = time.perf_counter()
    coord_result = coordinate_paths(raw_path_a, raw_path_b, coord_stats)
    t_coord_end = time.perf_counter()

    if coord_result is None:
        # Không thể phối hợp đường đi cố định
        _record_failure("coordination", {
            "search_a": {
                "expanded": result_a["expanded_states"],
                "runtime": result_a["runtime"],
                "path_length": len(raw_path_a) - 1,
            },
            "search_b": {
                "expanded": result_b["expanded_states"],
                "runtime": result_b["runtime"],
                "path_length": len(raw_path_b) - 1,
            },
            "coordination": {
                "expanded": coord_stats.get("expanded", 0),
                "runtime": t_coord_end - t_coord_start,
            },
        })
        return None

    new_path_a, new_path_b, coord_expanded = coord_result
    runtime = time.perf_counter() - t_start

    # Tạo joint path
    joint_path_list: List[JointState] = [
        (new_path_a[t], new_path_b[t]) for t in range(len(new_path_a))
    ]

    total_expanded = (
        result_a["expanded_states"]
        + result_b["expanded_states"]
        + coord_expanded
    )

    return {
        "joint_path": joint_path_list,
        "path_a": new_path_a,
        "path_b": new_path_b,
        "metrics": compute_metrics(
            algorithm_name="A* Độc lập",
            expanded_states=total_expanded,
            runtime=runtime,
            joint_path=joint_path_list,
            extra_info={
                "coordination_expanded": coord_expanded,
                "raw_path_length_a": len(raw_path_a) - 1,
                "raw_path_length_b": len(raw_path_b) - 1,
                "note": (
                    "Không đảm bảo tối ưu toàn cục. "
                    "Makespan có thể lớn hơn BFS/Joint A*."
                ),
            },
        ),
        "stage_metrics": {
            "search_a": {
                "expanded": result_a["expanded_states"],
                "runtime": result_a["runtime"],
                "path_length": len(raw_path_a) - 1,
            },
            "search_b": {
                "expanded": result_b["expanded_states"],
                "runtime": result_b["runtime"],
                "path_length": len(raw_path_b) - 1,
            },
            "coordination": {
                "expanded": coord_expanded,
                "runtime": t_coord_end - t_coord_start,
            },
        },
    }
