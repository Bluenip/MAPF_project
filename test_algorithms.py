"""
test_algorithms.py – Bộ kiểm thử cho ba thuật toán định tuyến hai robot.

Chạy: python3 -m unittest test_algorithms -v
Hoặc: python3 test_algorithms.py
Tổng cộng 48 bài kiểm thử (slide 16).

Các nhóm kiểm thử:
- TestValidation: kiểm tra đầu vào không hợp lệ.
- TestTrivialCase: cả hai robot đã ở đích.
- TestBFS: BFS tìm lời giải tối ưu, không xung đột.
- TestJointAStar: Joint A* tìm lời giải tối ưu, không xung đột.
- TestIndependentAStar: A* độc lập + phối hợp.
- TestOptimalEquality: BFS và Joint A* cho cùng makespan.
- TestObstacleNoSolution: không có lời giải.
- TestWaiting: robot phải đợi.
- TestPathValidator: kiểm tra bộ xác thực đường đi.
- TestUnschedulableIndependent: đường đi cố định không phối hợp được
  nhưng joint search có thể giải.
- TestCountActualMoves, TestManhattanDistance, TestMakespanHeuristic:
  các hàm tiện ích trong common.py.
"""

from __future__ import annotations

import unittest
from typing import List

from common import (
    Grid,
    JointState,
    count_actual_moves,
    has_edge_swap_conflict,
    manhattan_distance,
    makespan_heuristic,
    validate_joint_path,
    validate_individual_paths,
)
from bfs import bfs_joint
from astar import astar_independent, single_agent_astar
from joint_astar import joint_astar


# ─────────────────── Lưới thử nghiệm ──────────────────────────────────

def empty_grid(rows: int = 5, cols: int = 5) -> List[List[int]]:
    """Lưới trống (không chướng ngại vật)."""
    return [[0] * cols for _ in range(rows)]


def simple_obstacle_grid() -> List[List[int]]:
    """Lưới 5x5 với chướng ngại vật ở giữa.

       0 1 2 3 4
    0  . . . . .
    1  . # # # .
    2  . # . # .
    3  . # # # .
    4  . . . . .
    """
    g = empty_grid(5, 5)
    for y in range(1, 4):
        for x in range(1, 4):
            g[y][x] = 1
    g[2][2] = 0  # Lỗ hổng ở giữa
    return g


def corridor_grid() -> List[List[int]]:
    """Lưới 1x5 hành lang hẹp.

    . . . . .
    """
    return [[0] * 5]


def bay_corridor_grid() -> List[List[int]]:
    """Lưới hành lang có vịnh (bay) – dùng cho test unschedulable.

    Hai robot cần đi qua nhau trong hành lang hẹp.
    Có một "vịnh" (ô phụ) ở giữa cho phép tránh nhau.

       0 1 2 3 4
    0  # # . # #
    1  . . . . .
    2  # # . # #

    Hành lang hàng 1, vịnh ở (2,0) và (2,2) cho phép tránh.
    """
    g = [[1, 1, 0, 1, 1],
         [0, 0, 0, 0, 0],
         [1, 1, 0, 1, 1]]
    return g


def blocked_grid() -> List[List[int]]:
    """Lưới 3x3 bị chặn hoàn toàn.

    . # .
    # # #
    . # .
    """
    return [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]


# ─────────────────── Hàm tiện ích kiểm thử ────────────────────────────

def _check_collision_free(
    test_case: unittest.TestCase,
    joint_path: List[JointState],
) -> None:
    """Kiểm tra đường đi chung không có xung đột."""
    for t in range(len(joint_path)):
        pos_a, pos_b = joint_path[t]
        test_case.assertNotEqual(
            pos_a,
            pos_b,
            f"Xung đột đỉnh tại bước {t}: A={pos_a}, B={pos_b}",
        )
        if t > 0:
            prev_a, prev_b = joint_path[t - 1]
            test_case.assertFalse(
                has_edge_swap_conflict(prev_a, pos_a, prev_b, pos_b),
                f"Xung đột hoán đổi cạnh tại bước {t}",
            )


def _check_valid_moves(
    test_case: unittest.TestCase,
    grid: List[List[int]],
    joint_path: List[JointState],
) -> None:
    """Kiểm tra mọi bước di chuyển hợp lệ."""
    g = Grid(grid)
    for t in range(len(joint_path)):
        pos_a, pos_b = joint_path[t]
        test_case.assertTrue(
            g.is_free(pos_a[0], pos_a[1]),
            f"Bước {t}: A ở vị trí không hợp lệ {pos_a}",
        )
        test_case.assertTrue(
            g.is_free(pos_b[0], pos_b[1]),
            f"Bước {t}: B ở vị trí không hợp lệ {pos_b}",
        )
        if t > 0:
            prev_a, prev_b = joint_path[t - 1]
            for label, prev, curr in [("A", prev_a, pos_a), ("B", prev_b, pos_b)]:
                dist = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])
                test_case.assertLessEqual(
                    dist,
                    1,
                    f"Bước {t}: Robot {label} nhảy từ {prev} đến {curr}",
                )


# ══════════════════════════ KIỂM THỬ ══════════════════════════════════


class TestValidation(unittest.TestCase):
    """Kiểm tra xác thực đầu vào."""

    def test_out_of_bounds(self):
        """Vị trí ngoài biên lưới."""
        g = empty_grid(3, 3)
        with self.assertRaises(ValueError):
            bfs_joint(g, (5, 0), (2, 2), (0, 0), (1, 1))

    def test_obstacle_position(self):
        """Vị trí trên chướng ngại vật."""
        g = empty_grid(3, 3)
        g[1][1] = 1
        with self.assertRaises(ValueError):
            bfs_joint(g, (1, 1), (2, 2), (0, 0), (0, 2))

    def test_same_start(self):
        """Hai robot cùng vị trí bắt đầu."""
        g = empty_grid(3, 3)
        with self.assertRaises(ValueError):
            bfs_joint(g, (0, 0), (2, 2), (0, 0), (1, 1))

    def test_same_goal(self):
        """Hai robot cùng vị trí đích."""
        g = empty_grid(3, 3)
        with self.assertRaises(ValueError):
            bfs_joint(g, (0, 0), (2, 2), (1, 0), (2, 2))

    def test_invalid_input_astar(self):
        """Kiểm tra A* cũng xác thực đầu vào."""
        g = empty_grid(3, 3)
        with self.assertRaises(ValueError):
            astar_independent(g, (0, 0), (2, 2), (0, 0), (1, 1))

    def test_invalid_input_joint_astar(self):
        """Kiểm tra Joint A* cũng xác thực đầu vào."""
        g = empty_grid(3, 3)
        with self.assertRaises(ValueError):
            joint_astar(g, (10, 10), (2, 2), (0, 0), (1, 1))

    def test_non_rectangular_grid(self):
        """Lưới không phải hình chữ nhật (các hàng có độ dài khác nhau)."""
        g = [[0, 0, 0], [0, 0]]
        with self.assertRaises(ValueError):
            Grid(g)
        with self.assertRaises(ValueError):
            bfs_joint(g, (0, 0), (1, 1), (1, 0), (0, 1))

    def test_empty_grid_and_empty_rows(self):
        """Lưới rỗng hoặc chứa hàng rỗng."""
        with self.assertRaises(ValueError):
            Grid([])
        with self.assertRaises(ValueError):
            Grid([[]])
        with self.assertRaises(ValueError):
            Grid([[0, 0], []])

    def test_invalid_cell_values(self):
        """Giá trị ô không nằm trong {0, 1}."""
        with self.assertRaises(ValueError):
            Grid([[0, 2], [0, 0]])
        with self.assertRaises(ValueError):
            Grid([[0, -1], [0, 0]])
        with self.assertRaises(ValueError):
            Grid([[0, "0"], [0, 0]])
        with self.assertRaises(ValueError):
            Grid([[0, 0.5], [0, 0]])

    def test_coordinate_shape_invalid(self):
        """Toạ độ không đúng hình dạng (x, y) gồm 2 phần tử."""
        g = empty_grid(3, 3)
        with self.assertRaises(ValueError):
            bfs_joint(g, (0,), (2, 2), (0, 1), (1, 1))
        with self.assertRaises(ValueError):
            bfs_joint(g, (0, 0, 0), (2, 2), (0, 1), (1, 1))
        with self.assertRaises(ValueError):
            astar_independent(g, "0,0", (2, 2), (0, 1), (1, 1))
        for planner in (bfs_joint, astar_independent, joint_astar):
            with self.subTest(planner=planner.__name__):
                with self.assertRaises(ValueError):
                    planner(g, [0, 0], (2, 2), (0, 1), (1, 1))

    def test_coordinate_non_integer(self):
        """Toạ độ chứa giá trị không phải số nguyên."""
        g = empty_grid(3, 3)
        with self.assertRaises(ValueError):
            bfs_joint(g, (0.5, 1.0), (2, 2), (0, 1), (1, 1))
        with self.assertRaises(ValueError):
            joint_astar(g, ("0", "1"), (2, 2), (0, 1), (1, 1))
        with self.assertRaises(ValueError):
            astar_independent(g, (True, 0), (2, 2), (0, 1), (1, 1))


class TestTrivialCase(unittest.TestCase):
    """Trường hợp đặc biệt: cả hai robot đã ở đích."""

    def test_trivial_bfs(self):
        g = empty_grid(3, 3)
        result = bfs_joint(g, (0, 0), (0, 0), (1, 1), (1, 1))
        self.assertIsNotNone(result)
        self.assertEqual(result["metrics"].makespan, 0)
        self.assertEqual(len(result["joint_path"]), 1)

    def test_trivial_astar(self):
        g = empty_grid(3, 3)
        result = astar_independent(g, (0, 0), (0, 0), (1, 1), (1, 1))
        self.assertIsNotNone(result)
        self.assertEqual(result["metrics"].makespan, 0)

    def test_trivial_joint_astar(self):
        g = empty_grid(3, 3)
        result = joint_astar(g, (0, 0), (0, 0), (1, 1), (1, 1))
        self.assertIsNotNone(result)
        self.assertEqual(result["metrics"].makespan, 0)


class TestBFS(unittest.TestCase):
    """Kiểm thử BFS – lời giải tối ưu, không xung đột."""

    def test_simple_crossing(self):
        """Hai robot đi ngang qua nhau trên lưới trống."""
        g = empty_grid(3, 3)
        result = bfs_joint(g, (0, 0), (2, 0), (2, 0), (0, 0))
        # Đường đi ngang hàng 0, cần tránh nhau
        self.assertIsNotNone(result)
        _check_collision_free(self, result["joint_path"])
        _check_valid_moves(self, g, result["joint_path"])
        # Phải kết thúc đúng đích
        self.assertEqual(result["joint_path"][-1], ((2, 0), (0, 0)))

    def test_simple_non_crossing(self):
        """Hai robot đi cùng hướng, không xung đột."""
        g = empty_grid(5, 5)
        result = bfs_joint(g, (0, 0), (4, 0), (0, 4), (4, 4))
        self.assertIsNotNone(result)
        _check_collision_free(self, result["joint_path"])
        _check_valid_moves(self, g, result["joint_path"])
        self.assertEqual(result["metrics"].makespan, 4)

    def test_with_obstacles(self):
        """Lưới có chướng ngại vật."""
        g = simple_obstacle_grid()
        result = bfs_joint(g, (0, 0), (4, 4), (4, 0), (0, 4))
        self.assertIsNotNone(result)
        _check_collision_free(self, result["joint_path"])
        _check_valid_moves(self, g, result["joint_path"])

    def test_metrics_structure(self):
        """Kiểm tra cấu trúc metrics."""
        g = empty_grid(3, 3)
        result = bfs_joint(g, (0, 0), (2, 0), (0, 2), (2, 2))
        self.assertIsNotNone(result)
        m = result["metrics"]
        self.assertEqual(m.algorithm, "BFS (Joint)")
        self.assertGreater(m.makespan, 0)
        self.assertGreater(m.expanded_states, 0)
        self.assertGreater(m.runtime_seconds, 0)


class TestJointAStar(unittest.TestCase):
    """Kiểm thử Joint A* – tối ưu makespan."""

    def test_simple_crossing(self):
        """Hai robot đi ngang qua nhau."""
        g = empty_grid(3, 3)
        result = joint_astar(g, (0, 0), (2, 0), (2, 0), (0, 0))
        self.assertIsNotNone(result)
        _check_collision_free(self, result["joint_path"])
        _check_valid_moves(self, g, result["joint_path"])
        self.assertEqual(result["joint_path"][-1], ((2, 0), (0, 0)))

    def test_with_obstacles(self):
        """Lưới có chướng ngại vật."""
        g = simple_obstacle_grid()
        result = joint_astar(g, (0, 0), (4, 4), (4, 0), (0, 4))
        self.assertIsNotNone(result)
        _check_collision_free(self, result["joint_path"])
        _check_valid_moves(self, g, result["joint_path"])

    def test_fewer_expansions_than_bfs(self):
        """Joint A* mở rộng ít trạng thái hơn BFS (hoặc bằng)."""
        g = empty_grid(5, 5)
        r_bfs = bfs_joint(g, (0, 0), (4, 4), (4, 0), (0, 4))
        r_ja = joint_astar(g, (0, 0), (4, 4), (4, 0), (0, 4))
        self.assertIsNotNone(r_bfs)
        self.assertIsNotNone(r_ja)
        self.assertLessEqual(
            r_ja["metrics"].expanded_states,
            r_bfs["metrics"].expanded_states,
        )


class TestIndependentAStar(unittest.TestCase):
    """Kiểm thử A* độc lập + phối hợp."""

    def test_non_crossing(self):
        """Hai robot đi không giao nhau – phối hợp thành công."""
        g = empty_grid(5, 5)
        result = astar_independent(g, (0, 0), (4, 0), (0, 4), (4, 4))
        self.assertIsNotNone(result)
        _check_collision_free(self, result["joint_path"])
        _check_valid_moves(self, g, result["joint_path"])

    def test_crossing_with_coordination(self):
        """Hai robot giao nhau vuông góc – phối hợp bằng cách chèn bước đợi."""
        g = empty_grid(5, 5)
        start_a, goal_a = (0, 2), (4, 2)
        start_b, goal_b = (2, 0), (2, 4)
        result = astar_independent(g, start_a, goal_a, start_b, goal_b)
        self.assertIsNotNone(result)
        _check_collision_free(self, result["joint_path"])
        _check_valid_moves(self, g, result["joint_path"])
        # Do giao nhau tại (2, 2), robot phải đợi, makespan = 5 (> khoảng cách di chuyển thực tế 4)
        self.assertEqual(result["metrics"].makespan, 5)
        self.assertEqual(result["metrics"].path_length_a, 4)
        self.assertEqual(result["metrics"].path_length_b, 4)
        self.assertEqual(len(result["joint_path"]), 6)

    def test_stage_metrics(self):
        """Kiểm tra chi tiết metrics từng giai đoạn."""
        g = empty_grid(5, 5)
        result = astar_independent(g, (0, 0), (4, 0), (0, 4), (4, 4))
        self.assertIsNotNone(result)
        sm = result["stage_metrics"]
        self.assertIn("search_a", sm)
        self.assertIn("search_b", sm)
        self.assertIn("coordination", sm)
        self.assertGreater(sm["search_a"]["expanded"], 0)
        self.assertGreater(sm["search_b"]["expanded"], 0)

    def test_single_agent_astar_shortest(self):
        """A* đơn lẻ tìm đường ngắn nhất."""
        g = Grid(empty_grid(5, 5))
        result = single_agent_astar(g, (0, 0), (4, 4))
        self.assertIsNotNone(result)
        # Manhattan distance = 8, đường đi tối ưu = 9 vị trí
        self.assertEqual(len(result["path"]), 9)


class TestOptimalEquality(unittest.TestCase):
    """BFS và Joint A* phải cho cùng makespan (cả hai đều tối ưu)."""

    def test_equality_simple(self):
        g = empty_grid(5, 5)
        r_bfs = bfs_joint(g, (0, 0), (4, 4), (4, 0), (0, 4))
        r_ja = joint_astar(g, (0, 0), (4, 4), (4, 0), (0, 4))
        self.assertIsNotNone(r_bfs)
        self.assertIsNotNone(r_ja)
        self.assertEqual(
            r_bfs["metrics"].makespan,
            r_ja["metrics"].makespan,
            "BFS và Joint A* phải cho cùng makespan tối ưu.",
        )

    def test_equality_obstacles(self):
        g = simple_obstacle_grid()
        r_bfs = bfs_joint(g, (0, 0), (4, 4), (4, 0), (0, 4))
        r_ja = joint_astar(g, (0, 0), (4, 4), (4, 0), (0, 4))
        self.assertIsNotNone(r_bfs)
        self.assertIsNotNone(r_ja)
        self.assertEqual(
            r_bfs["metrics"].makespan,
            r_ja["metrics"].makespan,
        )

    def test_equality_crossing(self):
        """Hai robot phải đi qua nhau."""
        g = empty_grid(3, 3)
        r_bfs = bfs_joint(g, (0, 0), (2, 0), (2, 0), (0, 0))
        r_ja = joint_astar(g, (0, 0), (2, 0), (2, 0), (0, 0))
        self.assertIsNotNone(r_bfs)
        self.assertIsNotNone(r_ja)
        self.assertEqual(
            r_bfs["metrics"].makespan,
            r_ja["metrics"].makespan,
        )


class TestObstacleNoSolution(unittest.TestCase):
    """Không có lời giải vì bị chặn hoàn toàn."""

    def test_blocked_bfs(self):
        g = blocked_grid()
        result = bfs_joint(g, (0, 0), (2, 0), (0, 2), (2, 2))
        self.assertIsNone(result)

    def test_blocked_astar(self):
        g = blocked_grid()
        result = astar_independent(g, (0, 0), (2, 0), (0, 2), (2, 2))
        self.assertIsNone(result)

    def test_blocked_joint_astar(self):
        g = blocked_grid()
        result = joint_astar(g, (0, 0), (2, 0), (0, 2), (2, 2))
        self.assertIsNone(result)


class TestWaiting(unittest.TestCase):
    """Robot phải đợi để tránh xung đột."""

    def test_corridor_waiting(self):
        """Trong hành lang hẹp, robot phải đợi/tránh nhau.

        Lưới 3x3 corridor:
        . . .
        . . .
        . . .

        A: (0,0) → (2,0), B: (2,0) → (0,0)
        Robot phải tránh nhau – cần ít nhất 1 bước đợi.
        """
        g = empty_grid(3, 3)
        result = bfs_joint(g, (0, 0), (2, 0), (2, 0), (0, 0))
        self.assertIsNotNone(result)
        _check_collision_free(self, result["joint_path"])

        # Makespan phải > manhattan distance (=2) vì phải tránh nhau
        self.assertGreater(result["metrics"].makespan, 2)

    def test_tight_corridor_joint(self):
        """Hành lang chỉ 1 hàng – robot phải tránh qua vịnh."""
        g = bay_corridor_grid()
        # A: (0,1) → (4,1), B: (4,1) → (0,1)
        result = joint_astar(g, (0, 1), (4, 1), (4, 1), (0, 1))
        self.assertIsNotNone(result)
        _check_collision_free(self, result["joint_path"])
        _check_valid_moves(self, g, result["joint_path"])


class TestPathValidator(unittest.TestCase):
    """Kiểm tra bộ xác thực đường đi."""

    def test_valid_path(self):
        g = empty_grid(3, 3)
        result = bfs_joint(g, (0, 0), (2, 0), (0, 2), (2, 2))
        self.assertIsNotNone(result)
        # Không nên raise
        validate_joint_path(
            g, result["joint_path"],
            (0, 0), (2, 0), (0, 2), (2, 2),
        )

    def test_invalid_start(self):
        g = empty_grid(3, 3)
        path = [((1, 1), (2, 2)), ((2, 0), (2, 2))]
        with self.assertRaises(ValueError):
            validate_joint_path(g, path, (0, 0), (2, 0), (0, 2), (2, 2))

    def test_vertex_conflict_detected(self):
        g = empty_grid(3, 3)
        # Đường đi có xung đột đỉnh
        path = [((0, 0), (2, 0)), ((1, 0), (1, 0))]
        with self.assertRaises(ValueError):
            validate_joint_path(g, path, (0, 0), (1, 0), (2, 0), (1, 0))

    def test_individual_path_validation(self):
        g = empty_grid(3, 3)
        result = astar_independent(g, (0, 0), (2, 0), (0, 2), (2, 2))
        if result is not None:
            validate_individual_paths(
                g, result["path_a"], result["path_b"],
                (0, 0), (2, 0), (0, 2), (2, 2),
            )


class TestUnschedulableIndependent(unittest.TestCase):
    """Đường đi cố định A* không phối hợp được nhưng Joint A* vẫn giải được.

    Trong hành lang có vịnh: hai robot đi ngược nhau, đường đi ngắn nhất
    đều qua hành lang duy nhất → A* độc lập không phối hợp được (hoặc
    cho makespan kém hơn). Joint A* tìm được lời giải bằng đường vòng
    qua vịnh.
    """

    def _get_narrow_corridor_grid(self):
        """Hành lang 1 hàng, không có vịnh → không thể tránh nhau.

           0 1 2 3 4
        0  . . . . .

        A: (0,0) → (4,0), B: (4,0) → (0,0)
        Đường đi cố định trực tiếp xung đột, không có chỗ tránh.
        """
        return [[0, 0, 0, 0, 0]]

    def test_narrow_corridor_independent_fails(self):
        """A* độc lập thất bại trong hành lang 1 hàng."""
        g = self._get_narrow_corridor_grid()
        result = astar_independent(g, (0, 0), (4, 0), (4, 0), (0, 0))
        # Trong hành lang 1 hàng, không thể phối hợp (không có chỗ tránh)
        self.assertIsNone(result)

    def test_narrow_corridor_joint_also_fails(self):
        """Joint A* cũng thất bại trong hành lang 1 hàng
        (không có chỗ tránh)."""
        g = self._get_narrow_corridor_grid()
        result = joint_astar(g, (0, 0), (4, 0), (4, 0), (0, 0))
        self.assertIsNone(result)

    def test_bay_corridor_independent_vs_joint(self):
        """Hành lang có vịnh: A* độc lập thất bại nhưng Joint A* thành công.

        Lưới bay_corridor_grid:
           0 1 2 3 4
        0  # # . # #
        1  . . . . .
        2  # # . # #

        A: (0,1) → (4,1), B: (4,1) → (0,1)

        Đường đi ngắn nhất đơn lẻ: đều dọc hàng 1.
        Phối hợp đường đi cố định (dọc hàng 1) sẽ THẤT BẠI
        vì hai robot đi ngược chiều trong hành lang, cần tránh qua
        vịnh (2,0) hoặc (2,2) – nhưng coordinate_paths chỉ chèn đợi,
        không đi vòng.

        Joint A* tìm được lời giải: một robot tạm rẽ vào vịnh để
        nhường đường.
        """
        g = bay_corridor_grid()
        start_a, goal_a = (0, 1), (4, 1)
        start_b, goal_b = (4, 1), (0, 1)

        # A* độc lập: thất bại (không thể phối hợp đường đi cố định)
        result_indep = astar_independent(g, start_a, goal_a, start_b, goal_b)
        self.assertIsNone(
            result_indep,
            "A* độc lập phải thất bại: đường đi cố định trong "
            "hành lang không thể phối hợp bằng chèn đợi.",
        )

        # Joint A* (hoặc BFS): thành công vì có thể đi vòng qua vịnh
        result_joint = joint_astar(g, start_a, goal_a, start_b, goal_b)
        self.assertIsNotNone(
            result_joint,
            "Joint A* phải tìm được lời giải qua vịnh.",
        )
        _check_collision_free(self, result_joint["joint_path"])
        _check_valid_moves(self, g, result_joint["joint_path"])

        # BFS cũng phải thành công
        result_bfs = bfs_joint(g, start_a, goal_a, start_b, goal_b)
        self.assertIsNotNone(result_bfs)
        _check_collision_free(self, result_bfs["joint_path"])

        # BFS và Joint A* cùng makespan
        self.assertEqual(
            result_bfs["metrics"].makespan,
            result_joint["metrics"].makespan,
        )


class TestCountActualMoves(unittest.TestCase):
    """Kiểm tra đếm bước di chuyển thực tế."""

    def test_all_moves(self):
        positions = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(count_actual_moves(positions), 3)

    def test_with_waits(self):
        positions = [(0, 0), (0, 0), (1, 0), (1, 0), (2, 0)]
        self.assertEqual(count_actual_moves(positions), 2)

    def test_single_position(self):
        self.assertEqual(count_actual_moves([(0, 0)]), 0)


class TestManhattanDistance(unittest.TestCase):
    """Kiểm tra khoảng cách Manhattan."""

    def test_same_position(self):
        self.assertEqual(manhattan_distance((0, 0), (0, 0)), 0)

    def test_horizontal(self):
        self.assertEqual(manhattan_distance((0, 0), (5, 0)), 5)

    def test_diagonal(self):
        self.assertEqual(manhattan_distance((0, 0), (3, 4)), 7)


class TestMakespanHeuristic(unittest.TestCase):
    """Kiểm tra heuristic makespan."""

    def test_at_goals(self):
        self.assertEqual(
            makespan_heuristic(((1, 1), (2, 2)), (1, 1), (2, 2)), 0
        )

    def test_max_of_two(self):
        # A cách 3, B cách 5 → max = 5
        self.assertEqual(
            makespan_heuristic(((0, 0), (0, 0)), (2, 1), (3, 2)), 5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
