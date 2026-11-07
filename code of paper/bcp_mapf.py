"""Cài đặt lại thuật toán Branch-and-Cut-and-Price cho MAPF của
Lam, Le Bodic, Harabor & Stuckey, "Branch-and-Cut-and-Price for Multi-Agent Pathfinding", IJCAI 2019.

Hai biến thể như trong paper:
  * BCP-B (mục 3): master problem (1)-(3), cắt xung đột đỉnh (5) và cạnh (6),
                   pricer A* với chi phí bị phạt theo dual (7), rẽ nhánh theo cặp agent-đỉnh (3.6).
  * BCP   (mục 4): BCP-B + cắt corridor (8) + cắt rectangle symmetry (9) + rẽ nhánh theo độ dài đường đi (4.3).

Bộ giải LP là HiGHS (thay cho SCIP + CPLEX trong paper); phần branch-and-bound, sinh cột,
sinh lát cắt đều được viết lại bằng Python ở file này.

Quy ước: chi phí đường đi = thời điểm đến đích lần cuối (SOC chuẩn). Paper định nghĩa chi phí
là "số vị trí trong dãy" = SOC + 1 mỗi agent; hai cách chỉ lệch nhau một hằng số |A| nên lời
giải tối ưu như nhau.
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
import time
from collections import defaultdict

import highspy
import numpy as np

EPS = 1e-6
INF = highspy.kHighsInf
BIG_M = 1e6  # chi phí của biến nhân tạo (đảm bảo LP luôn khả thi)


class Timeout(Exception):
    pass


class Column:
    """Một đường đi p của agent a: path[t] là ô ở thời điểm t, path[-1] = đích."""

    __slots__ = ("idx", "agent", "path", "cost", "moves")

    def __init__(self, idx, agent, path):
        self.idx = idx
        self.agent = agent
        self.path = path
        self.cost = len(path) - 1
        # cạnh di chuyển (u, v, t_đến), bỏ qua hành động đợi
        self.moves = [(path[t - 1], path[t], t) for t in range(1, len(path)) if path[t - 1] != path[t]]

    def visits(self, cell, t) -> bool:
        return self.path[t] == cell if t < len(self.path) else cell == self.path[-1]


class Node:
    __slots__ = ("lb", "depth", "forbid", "require", "length")

    def __init__(self, lb, depth, forbid, require, length):
        self.lb = lb
        self.depth = depth
        self.forbid = forbid    # agent -> set((cell, t))   : agent không được đi qua đỉnh (cell, t)
        self.require = require  # agent -> {t: cell}        : agent bắt buộc ở cell tại thời điểm t
        self.length = length    # agent -> (lo, hi)         : giới hạn độ dài đường đi

    def child(self, **kw):
        n = Node(self.lb, self.depth + 1,
                 {a: set(s) for a, s in self.forbid.items()},
                 {a: dict(d) for a, d in self.require.items()},
                 dict(self.length))
        for k, v in kw.items():
            if k == "forbid":
                a, c, t = v
                n.forbid.setdefault(a, set()).add((c, t))
            elif k == "require":
                a, c, t = v
                n.require.setdefault(a, {})[t] = c
            elif k == "length":
                a, lo, hi = v
                olo, ohi = n.length.get(a, (0, math.inf))
                n.length[a] = (max(lo, olo), min(hi, ohi))
        return n

    def allows(self, col: Column) -> bool:
        a = col.agent
        lo, hi = self.length.get(a, (0, math.inf))
        if not lo <= col.cost <= hi:
            return False
        for c, t in self.forbid.get(a, ()):
            if col.visits(c, t):
                return False
        for t, c in self.require.get(a, {}).items():
            if not col.visits(c, t):
                return False
        return True


class BCPSolver:
    def __init__(self, grid, agents, variant="BCP", time_limit=300.0, verbose=False, horizon=None, incumbent=None):
        assert variant in ("BCP", "BCP-B")
        self.grid = grid
        self.agents = agents
        self.n = len(agents)
        self.variant = variant
        self.advanced = variant == "BCP"
        self.time_limit = time_limit
        self.verbose = verbose
        self.dist = [grid.bfs_distances(g) for _, g in agents]  # h(v) chính xác cho A*
        for a, (s, g) in enumerate(agents):
            if s not in self.dist[a]:
                raise ValueError(f"agent {a}: đích không tới được từ điểm xuất phát")
        # Giới hạn độ dài đường đi (xem hàm solve() ở cuối file để biết vì sao vẫn đảm bảo tối ưu).
        self.horizon = horizon if horizon is not None else math.inf
        self.incumbent = incumbent  # list đường đi khả thi ban đầu (từ heuristic) hoặc None

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        self.h = h
        self.n_rows = 0
        self.n_cols = 0
        self.cols: list[Column | None] = []  # chỉ số cột -> Column (None = biến nhân tạo)
        self.cols_of_agent = defaultdict(list)
        self.path_set = set()
        # chỉ mục để tính hệ số nhanh khi thêm hàng mới
        self.col_by_vertex = defaultdict(list)  # (cell, t) -> cột (t < len(path))
        self.col_by_goal = defaultdict(list)    # goal cell -> cột (đứng ở đích sau khi kết thúc)
        self.col_by_move = defaultdict(list)    # (u, v, t) -> cột
        # các hàng (ràng buộc)
        self.vertex_rows = {}                   # (cell, t) -> row          (5)
        self.goal_vertex_rows = defaultdict(list)  # cell -> [(t, row)]
        self.edge_rows = {}                     # (u, v, t) chuẩn hoá -> row (6)
        self.agent_edge_rows = defaultdict(list)  # (a, u, v, t) -> [row]   (8), (9)
        self.agent_row_terms = {}               # row -> [(a, u, v, t)]
        self.cut_keys = set()
        self.stats = defaultdict(int)

        # Ràng buộc (2): mỗi agent chọn ít nhất một đường đi, kèm biến nhân tạo chi phí BIG_M.
        for a in range(self.n):
            self._add_row(1.0, INF, [], [])
        self.artificial = []
        for a in range(self.n):
            self.artificial.append(self._add_raw_col(BIG_M, [a], [1.0]))
            self.cols.append(None)

    # ------------------------------------------------------------------ LP helpers
    def _add_row(self, lo, hi, idx, vals):
        self.h.addRow(lo, hi, len(idx), np.array(idx, dtype=np.int32), np.array(vals, dtype=np.float64))
        self.n_rows += 1
        return self.n_rows - 1

    def _add_raw_col(self, cost, idx, vals, ub=INF):
        self.h.addCol(cost, 0.0, ub, len(idx), np.array(idx, dtype=np.int32), np.array(vals, dtype=np.float64))
        self.n_cols += 1
        return self.n_cols - 1

    def add_column(self, agent, path):
        key = (agent, tuple(path))
        if key in self.path_set:
            return None
        self.path_set.add(key)
        col = Column(self.n_cols, agent, path)
        coef = defaultdict(float)
        coef[agent] += 1.0
        for t, c in enumerate(path):
            r = self.vertex_rows.get((c, t))
            if r is not None:
                coef[r] += 1
        goal = path[-1]
        for t, r in self.goal_vertex_rows.get(goal, ()):
            if t >= len(path):
                coef[r] += 1
        for u, v, t in col.moves:
            r = self.edge_rows.get(_ekey(u, v, t))
            if r is not None:
                coef[r] += 1
            for r in self.agent_edge_rows.get((agent, u, v, t), ()):
                coef[r] += 1
        idx = list(coef)
        self._add_raw_col(float(col.cost), idx, [coef[i] for i in idx])
        self.cols.append(col)
        self.cols_of_agent[agent].append(col)
        for t, c in enumerate(path):
            self.col_by_vertex[(c, t)].append(col)
        self.col_by_goal[goal].append(col)
        for m in col.moves:
            self.col_by_move[m].append(col)
        self.stats["columns"] += 1
        return col

    def add_vertex_row(self, cell, t):
        cols = [c for c in self.col_by_vertex.get((cell, t), ())]
        cols += [c for c in self.col_by_goal.get(cell, ()) if len(c.path) <= t]
        r = self._add_row(-INF, 1.0, [c.idx for c in cols], [1.0] * len(cols))
        self.vertex_rows[(cell, t)] = r
        self.goal_vertex_rows[cell].append((t, r))
        self.stats["vertex_cuts"] += 1

    def add_edge_row(self, key):
        u, v, t = key
        cols = self.col_by_move.get((u, v, t), []) + self.col_by_move.get((v, u, t), [])
        r = self._add_row(-INF, 1.0, [c.idx for c in cols], [1.0] * len(cols))
        self.edge_rows[key] = r
        self.stats["edge_cuts"] += 1

    def add_agent_edge_row(self, terms, rhs, kind):
        """Ràng buộc dạng sum_{(a,u,v,t) in terms} y^p_(u,v,t) λ_p <= rhs (dùng cho (8) và (9))."""
        coef = defaultdict(float)
        for a, u, v, t in terms:
            for c in self.col_by_move.get((u, v, t), ()):
                if c.agent == a:
                    coef[c.idx] += 1
        idx = list(coef)
        r = self._add_row(-INF, float(rhs), idx, [coef[i] for i in idx])
        for term in terms:
            self.agent_edge_rows[term].append(r)
        self.agent_row_terms[r] = terms
        self.stats[kind] += 1

    def apply_node_bounds(self, node: Node):
        idx, ub = [], []
        for col in self.cols:
            if col is None:
                continue
            idx.append(col.idx)
            ub.append(INF if col.cost <= self.horizon and node.allows(col) else 0.0)
        if idx:
            k = len(idx)
            self.h.changeColsBounds(k, np.array(idx, dtype=np.int32), np.zeros(k), np.array(ub, dtype=np.float64))

    def solve_lp(self):
        self.h.run()
        self.stats["lp_solves"] += 1
        status = self.h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"LP không giải được: {self.h.modelStatusToString(status)}")
        sol = self.h.getSolution()
        return self.h.getInfo().objective_function_value, list(sol.col_value), list(sol.row_dual)

    # ------------------------------------------------------------------ pricing (mục 3.5)
    def penalties(self, duals):
        vpen, epen = {}, {}
        apen = defaultdict(dict)
        for key, r in self.vertex_rows.items():
            if duals[r] < -EPS:
                vpen[key] = -duals[r]
        for key, r in self.edge_rows.items():
            if duals[r] < -EPS:
                epen[key] = -duals[r]
        for r, terms in self.agent_row_terms.items():
            if duals[r] < -EPS:
                for a, u, v, t in terms:
                    d = apen[a]
                    d[(u, v, t)] = d.get((u, v, t), 0.0) - duals[r]
        return vpen, epen, apen

    def price(self, a, alpha, vpen, epen, apen_a, node: Node):
        """A* trên đồ thị mở rộng theo thời gian, chi phí mỗi bước = 1 + các khoản phạt -β, -γ, -π, -ρ.
        Trả về (ĉ_p, path) của đường đi rẻ nhất, hoặc None nếu không có đường nào có ĉ_p < α_a."""
        s, goal = self.agents[a]
        dist = self.dist[a]
        forbid = node.forbid.get(a, set())
        require = node.require.get(a, {})
        lo, hi = node.length.get(a, (0, math.inf))
        hi = min(hi, self.horizon)

        # Sau thời điểm H không còn khoản phạt / ràng buộc nhánh nào: phần còn lại đi đường ngắn nhất.
        H = lo
        for (_, t) in vpen:
            H = max(H, t)
        for (_, _, t) in epen:
            H = max(H, t)
        for (_, _, t) in apen_a:
            H = max(H, t)
        for (_, t) in forbid:
            H = max(H, t)
        for t in require:
            H = max(H, t)

        # Agent tiếp tục chiếm đích sau khi kết thúc: cộng phạt -β tại (goal, t') với mọi t' > T.
        goal_pen = sorted((t, p) for (c, t), p in vpen.items() if c == goal)

        def tail(T):
            return sum(p for t, p in goal_pen if t > T)

        min_end = lo
        for (c, t) in forbid:
            if c == goal:
                min_end = max(min_end, t + 1)
        for t, c in require.items():
            if c != goal:
                min_end = max(min_end, t + 1)

        def ok(c, t):
            if (c, t) in forbid:
                return False
            r = require.get(t)
            return r is None or r == c

        if not ok(s, 0):
            return None
        bound = alpha - EPS  # chỉ quan tâm đường có ĉ_p < α_a
        best, best_end = math.inf, None
        g0 = vpen.get((s, 0), 0.0)
        gbest = {(s, 0): g0}
        parent = {(s, 0): None}
        heap = [(g0 + dist[s], 0, g0, s)]
        counter = 0
        while heap:
            f, negt, g, c = heapq.heappop(heap)
            t = -negt
            if f >= min(best, bound) - 1e-9:
                break
            if g > gbest.get((c, t), math.inf) + 1e-12:
                continue
            counter += 1
            if counter % 4096 == 0:
                self.check_time()
            if c == goal and min_end <= t <= hi:
                cand = g + tail(t)
                if cand < best - 1e-9:
                    best, best_end = cand, ("goal", c, t)
            if t >= H:
                d = dist[c]
                if t + d <= hi and t + d >= min_end:
                    cand = g + d
                    if cand < best - 1e-9:
                        best, best_end = cand, ("free", c, t)
                continue
            t1 = t + 1
            nbrs = list(self.grid.neighbors(c))
            nbrs.append(c)
            for n in nbrs:
                dn = dist.get(n)
                if dn is None or t1 + dn > hi or not ok(n, t1):
                    continue
                step = 1.0 + vpen.get((n, t1), 0.0)
                if n != c:
                    step += epen.get(_ekey(c, n, t1), 0.0) + apen_a.get((c, n, t1), 0.0)
                g1 = g + step
                if g1 + dn >= min(best, bound) - 1e-9:
                    continue
                if g1 < gbest.get((n, t1), math.inf) - 1e-12:
                    gbest[(n, t1)] = g1
                    parent[(n, t1)] = (c, t)
                    heapq.heappush(heap, (g1 + dn, -t1, g1, n))
        self.stats["pricer_calls"] += 1
        if best_end is None or best >= bound:
            return None
        kind, c, t = best_end
        path = []
        state = (c, t)
        while state is not None:
            path.append(state[0])
            state = parent[state]
        path.reverse()
        if kind == "free":  # nối đường ngắn nhất (không phạt) từ c tới đích
            while path[-1] != goal:
                cur = path[-1]
                path.append(next(n for n in self.grid.neighbors(cur) if dist.get(n) == dist[cur] - 1))
        return best, path

    # ------------------------------------------------------------------ separation (mục 3.3, 3.4, 4.1, 4.2)
    def separate(self, active):
        added = 0
        Tmax = max(len(c.path) for c, _ in active)
        x = defaultdict(float)
        for c, val in active:
            for t in range(Tmax + 1):
                x[(c.path[t] if t < len(c.path) else c.path[-1], t)] += val
        for key, val in x.items():
            if val > 1 + EPS and key not in self.vertex_rows:
                self.add_vertex_row(*key)
                added += 1
        y = defaultdict(float)
        for c, val in active:
            for u, v, t in c.moves:
                y[_ekey(u, v, t)] += val
        for key, val in y.items():
            if val > 1 + EPS and key not in self.edge_rows:
                self.add_edge_row(key)
                added += 1
        if added or not self.advanced:
            return added
        # Chỉ thêm lát cắt tăng cường (8), (9) khi các ràng buộc bài toán (5), (6) đã thoả.
        use = defaultdict(float)  # (a, u, v, t) -> tổng λ
        by_move = defaultdict(set)  # (u, v, t) -> các agent dùng cạnh
        for c, val in active:
            for u, v, t in c.moves:
                use[(c.agent, u, v, t)] += val
                by_move[(u, v, t)].add(c.agent)
        added += self.separate_corridor(use, by_move)
        added += self.separate_rectangle(active, use)
        return added

    def separate_corridor(self, use, by_move):
        """Ràng buộc (8): a1 dùng (l1,t)->(l2,t+1) hoặc (l1,t+1)->(l2,t+2), a2 dùng chiều ngược lại; tổng <= 1."""
        added = 0
        cands = set()
        for (a, u, v, T) in use:
            for t in (T - 1, T - 2):
                if t >= 0:
                    cands.add((a, u, v, t))
        for a1, l1, l2, t in cands:
            s1 = use.get((a1, l1, l2, t + 1), 0.0) + use.get((a1, l1, l2, t + 2), 0.0)
            others = by_move.get((l2, l1, t + 1), set()) | by_move.get((l2, l1, t + 2), set())
            for a2 in others:
                if a2 == a1:
                    continue
                s2 = use.get((a2, l2, l1, t + 1), 0.0) + use.get((a2, l2, l1, t + 2), 0.0)
                if s1 + s2 > 1 + EPS:
                    key = ("corr",) + min((a1, a2, l1, l2), (a2, a1, l2, l1)) + (t,)
                    if key in self.cut_keys:
                        continue
                    self.cut_keys.add(key)
                    terms = [(a1, l1, l2, t + 1), (a1, l1, l2, t + 2), (a2, l2, l1, t + 1), (a2, l2, l1, t + 2)]
                    self.add_agent_edge_row(terms, 1, "corridor_cuts")
                    added += 1
        return added

    def separate_rectangle(self, active, use):
        """Ràng buộc (9). Với hướng chéo (dx, dy), một đoạn đường đi "không trễ" theo hướng đó giữ nguyên
        bất biến K = t - dx*x - dy*y. Nếu agent H đi ngang qua hình chữ nhật (vào cạnh trái, ra cạnh phải)
        và agent V đi dọc qua (vào cạnh trên, ra cạnh dưới) cùng bất biến K thì chắc chắn có xung đột đỉnh,
        nên trong 4 nhóm cạnh vào/ra chỉ dùng được tối đa 3."""
        added = 0
        occ = defaultdict(list)
        for c, val in active:
            for t in range(1, len(c.path) - 1):
                occ[(c.path[t], t)].append(c)
        for (cell, t), cols in occ.items():
            if len({c.agent for c in cols}) < 2:
                continue
            for p1, p2 in itertools.combinations(cols, 2):
                if p1.agent == p2.agent:
                    continue
                for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    seg1 = _diag_segment(p1.path, t, dx, dy)
                    seg2 = _diag_segment(p2.path, t, dx, dy)
                    if seg1 is None or seg2 is None:
                        continue
                    for (H, segH), (V, segV) in (((p1, seg1), (p2, seg2)), ((p2, seg2), (p1, seg1))):
                        cut = self._rectangle(cell, t, dx, dy, H, segH, V, segV)
                        if cut is None:
                            continue
                        key, terms = cut
                        if key in self.cut_keys:
                            continue
                        lhs = sum(use.get(term, 0.0) for term in terms)
                        if lhs > 3 + EPS:
                            self.cut_keys.add(key)
                            self.add_agent_edge_row(terms, 3, "rectangle_cuts")
                            added += 1
        return added

    def _rectangle(self, cell, t, dx, dy, H, segH, V, segV):
        tr = lambda c: (dx * c[0], dy * c[1])  # đổi hệ toạ độ để hướng chéo luôn là (+1, +1)
        cX, cY = tr(cell)
        K = t - cX - cY
        # các bước đi ngang của H và đi dọc của V trong đoạn chéo (toạ độ đã đổi)
        h_entry, h_exit, v_entry, v_exit = {}, {}, {}, {}
        for i in range(segH[0], segH[1]):
            (X, Y), (X2, Y2) = tr(H.path[i]), tr(H.path[i + 1])
            if X2 == X + 1:
                if X2 <= cX:
                    h_entry[X2] = Y
                if X >= cX:
                    h_exit[X] = Y
        for i in range(segV[0], segV[1]):
            (X, Y), (X2, Y2) = tr(V.path[i]), tr(V.path[i + 1])
            if Y2 == Y + 1:
                if Y2 <= cY:
                    v_entry[Y2] = X
                if Y >= cY:
                    v_exit[Y] = X
        best_in = None
        for X0, yh in h_entry.items():
            for Y0, xv in v_entry.items():
                if yh >= Y0 and xv >= X0:
                    score = (cX - X0) + (cY - Y0)
                    if best_in is None or score > best_in[0]:
                        best_in = (score, X0, Y0)
        best_out = None
        for X1, yh in h_exit.items():
            for Y1, xv in v_exit.items():
                if yh <= Y1 and xv <= X1:
                    score = (X1 - cX) + (Y1 - cY)
                    if best_out is None or score > best_out[0]:
                        best_out = (score, X1, Y1)
        if best_in is None or best_out is None:
            return None
        _, X0, Y0 = best_in
        _, X1, Y1 = best_out
        back = lambda X, Y: (dx * X, dy * Y)
        g = self.grid
        terms = []

        def edge(agent, a, b, t_arr):
            u, v = back(*a), back(*b)
            if t_arr >= 1 and g.passable(u) and g.passable(v):
                terms.append((agent, u, v, t_arr))

        for Y in range(Y0, Y1 + 1):
            edge(H.agent, (X0 - 1, Y), (X0, Y), K + X0 + Y)
            edge(H.agent, (X1, Y), (X1 + 1, Y), K + X1 + Y + 1)
        for X in range(X0, X1 + 1):
            edge(V.agent, (X, Y0 - 1), (X, Y0), K + X + Y0)
            edge(V.agent, (X, Y1), (X, Y1 + 1), K + X + Y1 + 1)
        key = ("rect", H.agent, V.agent, dx, dy, X0, X1, Y0, Y1, K)
        return key, terms

    # ------------------------------------------------------------------ branching (mục 3.6, 4.3)
    def branch(self, node, active):
        if self.advanced:
            costs = defaultdict(set)
            for c, _ in active:
                costs[c.agent].add(c.cost)
            S = [a for a, cs in costs.items() if len(cs) > 1]
            if S:
                c = min((c for c, _ in active if c.agent in S), key=lambda c: (c.cost, c.agent))
                self.stats["length_branches"] += 1
                return [node.child(length=(c.agent, 0, c.cost)),
                        node.child(length=(c.agent, c.cost + 1, math.inf))]
        Tmax = max(len(c.path) for c, _ in active)
        x = defaultdict(float)
        users = defaultdict(set)
        per_agent = defaultdict(float)
        for c, val in active:
            for t in range(Tmax + 1):
                v = (c.path[t] if t < len(c.path) else c.path[-1], t)
                x[v] += val
                users[v].add(c.agent)
                per_agent[(c.agent,) + v] += val
        cands = [v for v, val in x.items() if EPS < val < 1 - EPS and len(users[v]) >= 2]
        if cands:
            v = min(cands, key=lambda v: (v[1], v[0]))
            mincost = {}
            for c, _ in active:
                if c.agent in users[v]:
                    mincost[c.agent] = min(mincost.get(c.agent, math.inf), c.cost)
            a = min(mincost, key=lambda a: (mincost[a], a))
        else:  # dự phòng: một agent tự dùng phân số một đỉnh
            fr = [k for k, val in per_agent.items() if EPS < val < 1 - EPS]
            a, cell, t = min(fr, key=lambda k: (k[2], k[0]))
            v = (cell, t)
        self.stats["vertex_branches"] += 1
        return [node.child(require=(a, v[0], v[1])), node.child(forbid=(a, v[0], v[1]))]

    # ------------------------------------------------------------------ main loop
    def check_time(self):
        if time.perf_counter() - self.t0 > self.time_limit:
            raise Timeout()

    def process_node(self, node: Node, ub):
        """Sinh cột + sinh lát cắt cho tới khi hội tụ. Trả về (lb, active, kind)."""
        self.apply_node_bounds(node)
        while True:
            self.check_time()
            obj, val, dual = self.solve_lp()
            vpen, epen, apen = self.penalties(dual)
            new = 0
            for a in range(self.n):
                res = self.price(a, dual[a], vpen, epen, apen.get(a, {}), node)
                if res is not None:
                    if self.add_column(a, res[1]) is not None:
                        new += 1
            if new:
                continue
            lb = obj
            if any(val[i] > EPS for i in self.artificial):
                return math.inf, None, "infeasible"
            if math.ceil(lb - EPS) >= ub:
                return lb, None, "pruned"
            active = [(c, val[c.idx]) for c in self.cols if c is not None and val[c.idx] > EPS]
            if self.separate(active):
                continue
            if all(v > 1 - EPS for _, v in active) and len({c.agent for c, _ in active}) == len(active) == self.n:
                return lb, active, "integral"
            return lb, active, "fractional"

    def solve(self):
        self.t0 = time.perf_counter()
        for a, (s, g) in enumerate(self.agents):  # cột khởi tạo: đường đi ngắn nhất của từng agent
            path = [s]
            while path[-1] != g:
                cur = path[-1]
                path.append(next(n for n in self.grid.neighbors(cur) if self.dist[a].get(n) == self.dist[a][cur] - 1))
            self.add_column(a, path)
        ub, best = math.inf, None
        if self.incumbent is not None:
            best = [self.add_column(a, p) or next(c for c in self.cols_of_agent[a] if c.path == p)
                    for a, p in enumerate(self.incumbent)]
            ub = sum(c.cost for c in best)
        root_lb = None
        global_lb = 0.0
        tie = itertools.count()
        open_nodes = [(0.0, 0, next(tie), Node(0.0, 0, {}, {}, {}))]
        status = "optimal"
        try:
            while open_nodes:
                lb0, _, _, node = heapq.heappop(open_nodes)
                global_lb = lb0
                if math.ceil(lb0 - EPS) >= ub:
                    continue
                self.stats["nodes"] += 1
                lb, active, kind = self.process_node(node, ub)
                if root_lb is None:
                    root_lb = lb
                    if self.verbose:
                        print(f"  root LB = {lb:.3f}  ({time.perf_counter() - self.t0:.2f}s)")
                if kind == "integral":
                    cost = round(sum(c.cost for c, _ in active))
                    if cost < ub:
                        ub = cost
                        best = sorted((c for c, _ in active), key=lambda c: c.agent)
                        if self.verbose:
                            print(f"  incumbent = {ub} tại nút {self.stats['nodes']}")
                elif kind == "fractional":
                    for ch in self.branch(node, active):
                        ch.lb = lb
                        heapq.heappush(open_nodes, (lb, -ch.depth, next(tie), ch))
            global_lb = ub if best is not None else math.inf
            if best is None:
                status = "infeasible"
        except Timeout:
            status = "timeout"
            if open_nodes:
                global_lb = min(global_lb, min(n[0] for n in open_nodes))
        elapsed = time.perf_counter() - self.t0
        return {
            "status": status,
            "soc": ub if best is not None else None,
            "lb": math.ceil(global_lb - EPS) if math.isfinite(global_lb) else None,
            "root_lb": root_lb,
            "paths": [c.path for c in best] if best else None,
            "time": elapsed,
            **{k: v for k, v in self.stats.items()},
        }


def _ekey(u, v, t):
    return (u, v, t) if u <= v else (v, u, t)


def _diag_segment(path, t, dx, dy):
    """Đoạn [i0, i1] lớn nhất chứa t mà mọi bước đều là (dx, 0) hoặc (0, dy) (không đợi)."""
    good = lambda i: (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) in ((dx, 0), (0, dy))
    i0 = t
    while i0 > 0 and good(i0 - 1):
        i0 -= 1
    i1 = t
    while i1 < len(path) - 1 and good(i1):
        i1 += 1
    if i0 == i1:
        return None
    return i0, i1


def prioritized_planning(grid, agents, dists, order, horizon, deadline):
    """Heuristic tìm nghiệm khả thi ban đầu (cận trên): lập kế hoạch lần lượt từng agent bằng A* không-thời gian,
    tránh các đỉnh/cạnh đã được agent trước giữ chỗ (kể cả ô đích bị chiếm vĩnh viễn sau khi tới)."""
    vres, eres = set(), set()
    parked = {}   # cell -> thời điểm agent trước bắt đầu đứng yên ở đó mãi mãi
    last_use = defaultdict(int)  # cell -> thời điểm cuối cùng có agent trước đi qua
    paths = [None] * len(agents)
    for a in order:
        s, g = agents[a]
        dist = dists[a]
        free = lambda c, t: (c, t) not in vres and not (c in parked and t >= parked[c])
        if not free(s, 0):
            return None
        heap = [(dist[s], 0, s)]
        parent = {(s, 0): None}
        end = None
        n_exp = 0
        while heap:
            f, t, c = heapq.heappop(heap)
            n_exp += 1
            if n_exp % 4096 == 0 and time.perf_counter() > deadline:
                return None
            if c == g and t >= last_use[g]:
                end = (c, t)
                break
            if t >= horizon:
                continue
            for n in list(grid.neighbors(c)) + [c]:
                st = (n, t + 1)
                if st in parent or n not in dist or not free(n, t + 1):
                    continue
                if n != c and (n, c, t + 1) in eres:
                    continue
                parent[st] = (c, t)
                heapq.heappush(heap, (t + 1 + dist[n], t + 1, n))
        if end is None:
            return None
        path, st = [], end
        while st is not None:
            path.append(st[0])
            st = parent[st]
        path.reverse()
        for t, c in enumerate(path):
            vres.add((c, t))
            last_use[c] = max(last_use[c], t + 1)
            if t:
                eres.add((path[t - 1], c, t))
        parked[g] = len(path) - 1
        paths[a] = path
    return paths


def solve(grid, agents, variant="BCP", time_limit=300.0, verbose=False):
    """Giải tối ưu (theo SOC) một instance.

    Vì đồ thị mở rộng theo thời gian là vô hạn, một nút nhánh vô nghiệm có thể khiến LP liên tục tìm ra
    đường đi muộn hơn một bước rồi lại bị lát cắt loại bỏ. Để thuật toán luôn dừng, độ dài mọi đường đi
    bị chặn bởi horizon T. Tính đúng đắn: nếu có nghiệm khả thi chi phí UB thì trong mọi nghiệm tối ưu,
    đường đi của agent a dài không quá UB - sum_{b != a} dist_b, nên chọn T = max_a của đại lượng đó
    là không làm mất nghiệm tối ưu. Nghiệm khả thi ban đầu lấy từ Prioritized Planning; nếu heuristic
    thất bại thì giải với T nhỏ rồi tăng dần (gấp đôi) cho tới khi có nghiệm, sau đó giải lại với T chính xác.
    """
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    dists = [grid.bfs_distances(g) for _, g in agents]
    for a, (s, _) in enumerate(agents):
        if s not in dists[a]:
            raise ValueError(f"agent {a}: đích không tới được từ điểm xuất phát")
    d = [dists[a][s] for a, (s, _) in enumerate(agents)]
    total = sum(d)
    exact_horizon = lambda ub: max(ub - (total - d[a]) for a in range(len(agents)))

    incumbent, pp_time = None, 0.0
    # Ngẫu nhiên có seed cố định (0): chỉ dùng để xáo thứ tự ưu tiên agent khi chạy lại Prioritized Planning
    # (tìm cận trên ban đầu). Seed cố định => cùng input luôn cho cùng thứ tự, cùng cận trên. Kết quả SOC
    # tối ưu không phụ thuộc vào thứ tự này (chỉ ảnh hưởng tốc độ).
    rng = random.Random(0)
    order = sorted(range(len(agents)), key=lambda a: -d[a])
    pp_deadline = t0 + 0.1 * time_limit
    for attempt in range(20):
        if time.perf_counter() > pp_deadline:
            break
        p = prioritized_planning(grid, agents, dists, order, max(d) + 4 * len(agents) + 50, pp_deadline)
        if p is not None and (incumbent is None or sum(len(x) - 1 for x in p) < sum(len(x) - 1 for x in incumbent)):
            incumbent = p
        order = rng.sample(range(len(agents)), len(agents))
    pp_time = time.perf_counter() - t0
    if verbose:
        print(f"  prioritized planning: UB = {sum(len(x) - 1 for x in incumbent) if incumbent else None} ({pp_time:.2f}s)")

    horizon = exact_horizon(sum(len(x) - 1 for x in incumbent)) if incumbent else max(d) + len(agents)
    totals = defaultdict(int)
    while True:
        remaining = deadline - time.perf_counter()
        exact = incumbent is not None
        r = BCPSolver(grid, agents, variant, max(remaining, 0.0), verbose, horizon, incumbent).solve()
        for k, v in r.items():
            if isinstance(v, int) and not isinstance(v, bool) and k not in ("soc", "lb"):
                totals[k] += v
        if r["status"] == "timeout" or exact:
            break
        if r["paths"] is not None:  # có nghiệm trong horizon tạm: giải lại với horizon chính xác
            incumbent = r["paths"]
            new_h = exact_horizon(r["soc"])
            if new_h <= horizon:
                break
            horizon = new_h
        else:
            horizon *= 2
    r.update(totals)
    if not exact and r["status"] == "timeout":
        r["lb"] = None  # cận dưới chỉ hợp lệ trong horizon tạm thời
    if r["paths"] is None and incumbent is not None:
        r["paths"], r["soc"] = incumbent, sum(len(x) - 1 for x in incumbent)
    r["time"] = time.perf_counter() - t0
    r["pp_time"] = pp_time
    r["horizon"] = horizon
    return r
