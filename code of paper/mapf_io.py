"""Đọc / ghi bản đồ và instance MAPF (định dạng MovingAI), cùng bộ kiểm tra lời giải.

Quy ước toạ độ trong toàn bộ code: một ô là tuple (x, y) với x = cột, y = hàng,
truy cập lưới bằng grid[y][x]. Đây cũng là quy ước của file .scen MovingAI.
"""
from __future__ import annotations

import os
from collections import deque

PASSABLE = set(".GS")  # ký tự ô đi được theo chuẩn MovingAI

# 4 hướng di chuyển (Bắc, Nam, Tây, Đông) như mục 3.1 của paper; hành động "đợi" xử lý riêng.
MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))


class Grid:
    def __init__(self, rows: list[str], name: str = "map"):
        self.name = name
        self.height = len(rows)
        self.width = len(rows[0]) if rows else 0
        self.free = [[ch in PASSABLE for ch in row] for row in rows]

    def passable(self, c) -> bool:
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height and self.free[y][x]

    def neighbors(self, c):
        x, y = c
        for dx, dy in MOVES:
            n = (x + dx, y + dy)
            if self.passable(n):
                yield n

    def free_cells(self):
        return [(x, y) for y in range(self.height) for x in range(self.width) if self.free[y][x]]

    def bfs_distances(self, src) -> dict:
        """Khoảng cách 4-liên thông ngắn nhất từ src tới mọi ô đi được."""
        dist = {src: 0}
        q = deque([src])
        while q:
            c = q.popleft()
            d = dist[c] + 1
            for n in self.neighbors(c):
                if n not in dist:
                    dist[n] = d
                    q.append(n)
        return dist

    def largest_component(self) -> set:
        seen, best = set(), set()
        for c in self.free_cells():
            if c in seen:
                continue
            comp = set(self.bfs_distances(c))
            seen |= comp
            if len(comp) > len(best):
                best = comp
        return best


def load_map(path: str) -> Grid:
    with open(path) as f:
        lines = [l.rstrip("\n\r") for l in f]
    i = next(k for k, l in enumerate(lines) if l.strip() == "map")
    header = {l.split()[0]: l.split()[1] for l in lines[:i] if len(l.split()) == 2}
    h, w = int(header["height"]), int(header["width"])
    rows = [l[:w].ljust(w, "@") for l in lines[i + 1 : i + 1 + h]]
    return Grid(rows, os.path.splitext(os.path.basename(path))[0])


def save_map(path: str, rows: list[str]):
    with open(path, "w") as f:
        f.write(f"type octile\nheight {len(rows)}\nwidth {len(rows[0])}\nmap\n")
        f.write("\n".join(rows) + "\n")


def load_scen(path: str, agent_limit: int | None = None):
    """Trả về (tên file map, danh sách (start, goal))."""
    agents, map_name = [], None
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            map_name = parts[1]
            sx, sy, gx, gy = map(int, parts[4:8])
            agents.append(((sx, sy), (gx, gy)))
            if agent_limit and len(agents) >= agent_limit:
                break
    return map_name, agents


def save_scen(path: str, map_file: str, grid: Grid, agents):
    """Cột cuối (optimal length) chỉ được tính cho bản đồ nhỏ; với bản đồ lớn ghi 0
    vì việc chạy BFS cho hàng chục nghìn agent rất chậm, và không solver nào dùng cột này."""
    small = grid.width * grid.height <= 10_000
    with open(path, "w") as f:
        f.write("version 1\n")
        for s, g in agents:
            d = grid.bfs_distances(s)[g] if small else 0
            f.write(f"0\t{map_file}\t{grid.width}\t{grid.height}\t{s[0]}\t{s[1]}\t{g[0]}\t{g[1]}\t{d}\n")


def validate_solution(grid: Grid, agents, paths) -> list[str]:
    """Kiểm tra lời giải theo định nghĩa của paper: đường đi hợp lệ, không xung đột
    đỉnh (kể cả khi agent đã đứng ở đích) và không xung đột cạnh (hoán đổi vị trí).
    Trả về danh sách lỗi (rỗng = hợp lệ)."""
    errors = []
    if len(paths) != len(agents):
        return [f"số đường đi {len(paths)} != số agent {len(agents)}"]
    for a, ((s, g), p) in enumerate(zip(agents, paths)):
        if p[0] != s or p[-1] != g:
            errors.append(f"agent {a}: sai điểm đầu/cuối")
        for t in range(len(p) - 1):
            u, v = p[t], p[t + 1]
            if not grid.passable(v) or (u != v and abs(u[0] - v[0]) + abs(u[1] - v[1]) != 1):
                errors.append(f"agent {a}: bước không hợp lệ {u}->{v} tại t={t}")
    T = max(len(p) for p in paths)
    at = lambda p, t: p[t] if t < len(p) else p[-1]
    for t in range(T + 1):
        seen = {}
        for a, p in enumerate(paths):
            c = at(p, t)
            if c in seen:
                errors.append(f"xung đột đỉnh: agent {seen[c]} và {a} tại {c}, t={t}")
            seen[c] = a
        if t == 0:
            continue
        moves = {}
        for a, p in enumerate(paths):
            u, v = at(p, t - 1), at(p, t)
            if u != v:
                if (v, u) in moves:
                    errors.append(f"xung đột cạnh: agent {moves[(v, u)]} và {a} trên {u}<->{v}, t={t}")
                moves[(u, v)] = a
    return errors


def sum_of_costs(paths) -> int:
    """SOC chuẩn: tổng thời điểm đến đích lần cuối (= số bước, kể cả bước đợi)."""
    return sum(len(p) - 1 for p in paths)
