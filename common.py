"""
common.py – Tiện ích dùng chung cho bài toán định tuyến hai robot trong kho hàng.

Mô-đun này cung cấp:
- Lớp Grid: biểu diễn lưới hình chữ nhật (0 = trống, 1 = chướng ngại vật).
- Các hàm kiểm tra đầu vào (validate_inputs).
- Phát hiện xung đột (đỉnh và hoán đổi cạnh).
- Khoảng cách Manhattan và heuristic makespan.
- Dựng lại đường đi, tính toán metrics, hiển thị lưới.

Quy ước toạ độ: (x, y) với x = cột, y = hàng. grid[y][x].
Hướng di chuyển: Lên, Xuống, Trái, Phải, Đợi.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

# ──────────────────────────── Kiểu dữ liệu ────────────────────────────

Position = Tuple[int, int]          # (x, y)
JointState = Tuple[Position, Position]  # ((xA, yA), (xB, yB))

# 5 hướng di chuyển: dx, dy
DIRECTIONS: List[Tuple[int, int]] = [
    (0, -1),   # Lên    (y giảm)
    (0, 1),    # Xuống  (y tăng)
    (-1, 0),   # Trái   (x giảm)
    (1, 0),    # Phải   (x tăng)
    (0, 0),    # Đợi    (không di chuyển)
]


# ──────────────────────────── Lớp Grid ──────────────────────────────────

class Grid:
    """Lưới hình chữ nhật cho bài toán kho hàng.

    Attributes:
        data: Danh sách 2D, data[y][x] ∈ {0, 1}.
        rows: Số hàng.
        cols: Số cột.
    """

    def __init__(self, data: List[List[int]]) -> None:
        if not isinstance(data, (list, tuple)) or not data:
            raise ValueError("Lưới không được rỗng.")
        if not isinstance(data[0], (list, tuple)) or not data[0]:
            raise ValueError("Lưới không được rỗng.")
        first_len = len(data[0])
        for r, row in enumerate(data):
            if not isinstance(row, (list, tuple)) or not row:
                raise ValueError(f"Hàng {r} của lưới không hợp lệ hoặc rỗng.")
            if len(row) != first_len:
                raise ValueError(
                    f"Hàng {r} có độ dài {len(row)} khác với {first_len} "
                    f"(lưới phải là hình chữ nhật)."
                )
            for c, val in enumerate(row):
                if type(val) is not int or val not in (0, 1):
                    raise ValueError(
                        f"Giá trị ô ({c}, {r}) không hợp lệ: {val}. "
                        f"Chỉ chấp nhận 0 hoặc 1."
                    )
        self.data = data
        self.rows = len(data)
        self.cols = first_len

    def in_bounds(self, x: int, y: int) -> bool:
        """Kiểm tra (x, y) nằm trong biên lưới."""
        return 0 <= x < self.cols and 0 <= y < self.rows

    def is_free(self, x: int, y: int) -> bool:
        """Kiểm tra ô (x, y) trống (không phải chướng ngại vật)."""
        return self.in_bounds(x, y) and self.data[y][x] == 0

    def neighbors(self, x: int, y: int) -> List[Position]:
        """Trả về danh sách các vị trí lân cận hợp lệ (gồm cả đợi tại chỗ)."""
        result: List[Position] = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if self.is_free(nx, ny):
                result.append((nx, ny))
        return result


# ────────────────────────── Kiểm tra đầu vào ───────────────────────────

def validate_inputs(
    grid: List[List[int]],
    start_a: Position,
    goal_a: Position,
    start_b: Position,
    goal_b: Position,
) -> Grid:
    """Kiểm tra tính hợp lệ của đầu vào và trả về đối tượng Grid.

    Raises:
        ValueError: nếu đầu vào không hợp lệ (ngoài biên, chướng ngại vật,
                     vị trí trùng nhau, hoặc sai định dạng toạ độ).
    """
    g = Grid(grid)

    for label, pos in [
        ("start_a", start_a),
        ("goal_a", goal_a),
        ("start_b", start_b),
        ("goal_b", goal_b),
    ]:
        if not (isinstance(pos, tuple) and len(pos) == 2):
            raise ValueError(
                f"{label} = {pos} phải là tuple (x, y) gồm 2 phần tử."
            )
        x, y = pos
        if type(x) is not int or type(y) is not int:
            raise ValueError(
                f"{label} = {pos} toạ độ (x, y) phải là số nguyên."
            )
        if not g.in_bounds(x, y):
            raise ValueError(
                f"{label} = {pos} nằm ngoài biên lưới "
                f"({g.cols}x{g.rows})."
            )
        if not g.is_free(x, y):
            raise ValueError(
                f"{label} = {pos} là chướng ngại vật (grid[{y}][{x}] = 1)."
            )

    if start_a == start_b:
        raise ValueError(
            f"Hai robot không được có cùng vị trí bắt đầu: {start_a}."
        )
    if goal_a == goal_b:
        raise ValueError(
            f"Hai robot không được có cùng vị trí đích: {goal_a}."
        )

    return g


# ─────────────────────── Phát hiện xung đột ────────────────────────────

def has_vertex_conflict(pos_a: Position, pos_b: Position) -> bool:
    """Hai robot chiếm cùng một ô (xung đột đỉnh)."""
    return pos_a == pos_b


def has_edge_swap_conflict(
    prev_a: Position,
    curr_a: Position,
    prev_b: Position,
    curr_b: Position,
) -> bool:
    """Hai robot hoán đổi vị trí (xung đột hoán đổi cạnh)."""
    return prev_a == curr_b and prev_b == curr_a


# ─────────────────── Khoảng cách & Heuristic ───────────────────────────

def manhattan_distance(pos1: Position, pos2: Position) -> int:
    """Khoảng cách Manhattan giữa hai vị trí."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def makespan_heuristic(
    state: JointState,
    goal_a: Position,
    goal_b: Position,
) -> int:
    """Heuristic makespan = max(Manhattan_A, Manhattan_B).

    Đây là heuristic chấp nhận được (admissible) cho bài toán tối ưu makespan
    vì mỗi robot cần ít nhất Manhattan bước để tới đích.
    """
    return max(
        manhattan_distance(state[0], goal_a),
        manhattan_distance(state[1], goal_b),
    )


# ──────────────────── Dựng lại đường đi ────────────────────────────────

def reconstruct_path(
    parents: Dict[Any, Any],
    current: Any,
) -> List[Any]:
    """Dựng lại đường đi từ trạng thái bắt đầu đến current bằng bản đồ cha.

    Returns:
        Danh sách các trạng thái từ start đến current (bao gồm cả hai đầu).
    """
    path = [current]
    while current in parents:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


# ──────────────────────── Metrics ───────────────────────────────────────

class Metrics(NamedTuple):
    """Kết quả đo lường hiệu suất thuật toán.

    Attributes:
        algorithm: Tên thuật toán.
        makespan: Số bước đồng bộ (len(joint_path) - 1).
        path_length_a: Số bước robot A thực sự di chuyển (không tính đợi).
        path_length_b: Số bước robot B thực sự di chuyển (không tính đợi).
        expanded_states: Số trạng thái đã mở rộng.
        runtime_seconds: Thời gian chạy (giây).
        extra_info: Thông tin bổ sung (tuỳ chọn).
    """

    algorithm: str
    makespan: int
    path_length_a: int
    path_length_b: int
    expanded_states: int
    runtime_seconds: float
    extra_info: Dict[str, Any] = {}


def count_actual_moves(positions: Sequence[Position]) -> int:
    """Đếm số bước mà robot thực sự di chuyển (vị trí thay đổi).

    Args:
        positions: Chuỗi các vị trí (x, y) theo thời gian.

    Returns:
        Số bước mà vị trí khác với bước trước đó.
    """
    moves = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i - 1]:
            moves += 1
    return moves


def compute_metrics(
    algorithm_name: str,
    expanded_states: int,
    runtime: float,
    joint_path: Optional[List[JointState]] = None,
    path_a: Optional[List[Position]] = None,
    path_b: Optional[List[Position]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Metrics:
    """Tính toán metrics từ kết quả thuật toán.

    Cung cấp joint_path HOẶC (path_a, path_b).
    Nếu cung cấp joint_path, path_a/b sẽ được trích xuất tự động.
    Nếu cung cấp path_a/path_b, chúng sẽ được căn bằng (pad) về cùng độ dài
    và makespan = max(len_a, len_b) - 1.
    """
    if joint_path is not None:
        positions_a = [s[0] for s in joint_path]
        positions_b = [s[1] for s in joint_path]
        makespan = len(joint_path) - 1
    elif path_a is not None and path_b is not None:
        len_a, len_b = len(path_a), len(path_b)
        makespan = max(len_a, len_b) - 1
        # Pad đường đi ngắn hơn bằng vị trí cuối (robot đợi tại đích)
        if len_a < len_b:
            positions_a = list(path_a) + [path_a[-1]] * (len_b - len_a)
        else:
            positions_a = list(path_a)
        if len_b < len_a:
            positions_b = list(path_b) + [path_b[-1]] * (len_a - len_b)
        else:
            positions_b = list(path_b)
    else:
        raise ValueError("Phải cung cấp joint_path hoặc (path_a, path_b).")

    return Metrics(
        algorithm=algorithm_name,
        makespan=makespan,
        path_length_a=count_actual_moves(positions_a),
        path_length_b=count_actual_moves(positions_b),
        expanded_states=expanded_states,
        runtime_seconds=runtime,
        extra_info=extra_info or {},
    )


# ──────────── Xác thực đường đi (Path Validator) ───────────────────────

def validate_joint_path(
    grid: List[List[int]],
    path: List[JointState],
    start_a: Position,
    goal_a: Position,
    start_b: Position,
    goal_b: Position,
) -> bool:
    """Xác thực đường đi chung không có xung đột.

    Kiểm tra:
    - Bắt đầu đúng, kết thúc đúng.
    - Mỗi bước di chuyển hợp lệ (trong biên, ô trống, khoảng cách ≤ 1).
    - Không có xung đột đỉnh hay hoán đổi cạnh.

    Returns:
        True nếu đường đi hợp lệ.

    Raises:
        ValueError: nếu đường đi không hợp lệ (kèm mô tả lỗi).
    """
    if not path:
        raise ValueError("Đường đi rỗng.")

    g = Grid(grid)

    # Kiểm tra điểm bắt đầu
    if path[0] != (start_a, start_b):
        raise ValueError(
            f"Điểm bắt đầu không đúng: {path[0]} != "
            f"{(start_a, start_b)}."
        )

    # Kiểm tra điểm kết thúc
    if path[-1] != (goal_a, goal_b):
        raise ValueError(
            f"Điểm kết thúc không đúng: {path[-1]} != "
            f"{(goal_a, goal_b)}."
        )

    for t, (pos_a, pos_b) in enumerate(path):
        # Kiểm tra vị trí hợp lệ
        for label, pos in [("A", pos_a), ("B", pos_b)]:
            x, y = pos
            if not g.is_free(x, y):
                raise ValueError(
                    f"Bước {t}: Robot {label} ở vị trí không hợp lệ {pos}."
                )

        # Xung đột đỉnh
        if has_vertex_conflict(pos_a, pos_b):
            raise ValueError(
                f"Bước {t}: Xung đột đỉnh – cả hai robot ở {pos_a}."
            )

        # Kiểm tra di chuyển hợp lệ và xung đột cạnh
        if t > 0:
            prev_a, prev_b = path[t - 1]
            for label, prev, curr in [("A", prev_a, pos_a), ("B", prev_b, pos_b)]:
                dx = abs(curr[0] - prev[0])
                dy = abs(curr[1] - prev[1])
                if dx + dy > 1:
                    raise ValueError(
                        f"Bước {t}: Robot {label} di chuyển không hợp lệ "
                        f"từ {prev} đến {curr} (khoảng cách > 1)."
                    )

            if has_edge_swap_conflict(prev_a, pos_a, prev_b, pos_b):
                raise ValueError(
                    f"Bước {t}: Xung đột hoán đổi cạnh giữa "
                    f"A({prev_a}->{pos_a}) và B({prev_b}->{pos_b})."
                )

    return True


def validate_individual_paths(
    grid: List[List[int]],
    path_a: List[Position],
    path_b: List[Position],
    start_a: Position,
    goal_a: Position,
    start_b: Position,
    goal_b: Position,
) -> bool:
    """Xác thực hai đường đi riêng lẻ (sau khi căn bằng độ dài).

    Chuyển đổi thành joint path rồi gọi validate_joint_path.

    Returns:
        True nếu hợp lệ.

    Raises:
        ValueError: nếu không hợp lệ.
    """
    # Căn bằng độ dài
    len_a, len_b = len(path_a), len(path_b)
    padded_a = list(path_a)
    padded_b = list(path_b)
    if len_a < len_b:
        padded_a += [path_a[-1]] * (len_b - len_a)
    elif len_b < len_a:
        padded_b += [path_b[-1]] * (len_a - len_b)

    joint_path: List[JointState] = [
        (padded_a[t], padded_b[t]) for t in range(len(padded_a))
    ]

    return validate_joint_path(
        grid, joint_path, start_a, goal_a, start_b, goal_b
    )


# ───────────────────── Hiển thị lưới ────────────────────────────────────

def format_grid(
    grid: List[List[int]],
    positions_a: Optional[List[Position]] = None,
    positions_b: Optional[List[Position]] = None,
    start_a: Optional[Position] = None,
    goal_a: Optional[Position] = None,
    start_b: Optional[Position] = None,
    goal_b: Optional[Position] = None,
    step: Optional[int] = None,
) -> str:
    """Tạo chuỗi hiển thị lưới.

    Ký hiệu:
        .  = ô trống
        #  = chướng ngại vật
        A  = vị trí hiện tại của Robot A (tại bước `step`)
        B  = vị trí hiện tại của Robot B
        X  = cả hai robot cùng một ô (xung đột!)
        a  = điểm bắt đầu Robot A,  α = điểm đích Robot A
        b  = điểm bắt đầu Robot B,  β = điểm đích Robot B
        Khi một ô vừa là điểm bắt đầu vừa là điểm đích, ưu tiên hiển thị theo
        thứ tự a, α, b, β.
    """
    g = Grid(grid)
    # Xác định vị trí hiện tại của robot
    current_a: Optional[Position] = None
    current_b: Optional[Position] = None
    if positions_a and step is not None and 0 <= step < len(positions_a):
        current_a = positions_a[step]
    if positions_b and step is not None and 0 <= step < len(positions_b):
        current_b = positions_b[step]

    lines: List[str] = []
    # Header: column numbers
    header = "   " + " ".join(f"{c}" for c in range(g.cols))
    lines.append(header)

    for y in range(g.rows):
        row_str = f"{y:2d} "
        for x in range(g.cols):
            pos = (x, y)
            if current_a == pos and current_b == pos:
                row_str += "X "
            elif current_a == pos:
                row_str += "A "
            elif current_b == pos:
                row_str += "B "
            elif goal_a == pos and goal_b == pos:
                # Không nên xảy ra (đã validate), nhưng phòng trường hợp
                row_str += "G "
            elif start_a == pos:
                row_str += "a "
            elif goal_a == pos:
                row_str += "α "
            elif start_b == pos:
                row_str += "b "
            elif goal_b == pos:
                row_str += "β "
            elif grid[y][x] == 1:
                row_str += "# "
            else:
                row_str += ". "
        lines.append(row_str.rstrip())
    return "\n".join(lines)


def print_path_steps(
    grid: List[List[int]],
    joint_path: List[JointState],
    start_a: Position,
    goal_a: Position,
    start_b: Position,
    goal_b: Position,
    max_steps: Optional[int] = None,
) -> None:
    """In từng bước của đường đi chung."""
    positions_a = [s[0] for s in joint_path]
    positions_b = [s[1] for s in joint_path]
    n = len(joint_path)
    if max_steps is not None:
        n = min(n, max_steps)
    for t in range(n):
        print(f"\n--- Bước {t} ---")
        print(
            format_grid(
                grid,
                positions_a=positions_a,
                positions_b=positions_b,
                start_a=start_a,
                goal_a=goal_a,
                start_b=start_b,
                goal_b=goal_b,
                step=t,
            )
        )
