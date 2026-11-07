"""Kiểm thử tính đúng đắn: so SOC tối ưu của BCP và BCP-B với một bộ giải vét cạn
(Dijkstra trên không gian trạng thái chung) trên các instance nhỏ ngẫu nhiên.

Chạy:  .venv/bin/python test_bcp.py [số_instance]
"""
import heapq
import random
import sys

from bcp_mapf import solve
from mapf_io import Grid, sum_of_costs, validate_solution


def brute_force_soc(grid, agents):
    """Dijkstra trên trạng thái (vị trí các agent, số bước đã đứng ở đích của từng agent).
    Chi phí = tổng thời điểm đến đích lần cuối; agent rời đích phải "trả" lại các bước đã đứng đợi."""
    n = len(agents)
    goals = tuple(g for _, g in agents)
    start = (tuple(s for s, _ in agents), tuple(0 for _ in agents))
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, st = heapq.heappop(pq)
        if d > dist[st]:
            continue
        pos, waits = st
        if pos == goals:
            return d
        choices = [list(grid.neighbors(p)) + [p] for p in pos]

        # nw[i]: số bước agent i đã ở đích mà chưa bị tính phí (sẽ phải trả nếu rời đích)
        def rec2(i, new, cost, nw):
            if i == n:
                key = (tuple(new), tuple(nw))
                nd = d + cost
                if nd < dist.get(key, 1 << 30):
                    dist[key] = nd
                    heapq.heappush(pq, (nd, key))
                return
            for q in choices[i]:
                if q in new:
                    continue
                if any(new[j] == pos[i] and pos[j] == q for j in range(i)):
                    continue
                if q == goals[i] and pos[i] == goals[i]:  # đứng yên ở đích: hoãn tính phí
                    rec2(i + 1, new + [q], cost, nw + [waits[i] + 1])
                else:
                    rec2(i + 1, new + [q], cost + 1 + waits[i], nw + [0])

        rec2(0, [], 0, [])
    return None


def solvable(grid, agents):
    """BFS trên không gian vị trí chung (hữu hạn): có tồn tại lời giải hay không."""
    start = tuple(s for s, _ in agents)
    goal = tuple(g for _, g in agents)
    seen, frontier = {start}, [start]
    while frontier:
        nxt = []
        for pos in frontier:
            if pos == goal:
                return True
            stack = [((), 0)]
            while stack:
                new, i = stack.pop()
                if i == len(pos):
                    if new not in seen:
                        seen.add(new)
                        nxt.append(new)
                    continue
                for q in list(grid.neighbors(pos[i])) + [pos[i]]:
                    if q in new or any(new[j] == pos[i] and pos[j] == q for j in range(i)):
                        continue
                    stack.append((new + (q,), i + 1))
        frontier = nxt
    return False


def random_instance(rng, w, h, n_obs, k):
    cells = [(x, y) for y in range(h) for x in range(w)]
    while True:
        obs = set(rng.sample(cells, n_obs))
        rows = ["".join("@" if (x, y) in obs else "." for x in range(w)) for y in range(h)]
        g = Grid(rows)
        comp = sorted(g.largest_component())
        if len(comp) >= 2 * k + 2:
            agents = list(zip(rng.sample(comp, k), rng.sample(comp, k)))
            if solvable(g, agents):
                return g, agents


def main():
    n_tests = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    # Ngẫu nhiên có seed cố định (7): các instance kiểm thử (kích thước lưới, chướng ngại, start/goal)
    # được sinh ngẫu nhiên nhưng luôn giống hệt nhau mỗi lần chạy, nên kết quả "60/60" tái lập được.
    rng = random.Random(7)
    ok = 0
    for i in range(n_tests):
        w, h = rng.choice([(3, 3), (4, 3), (4, 4), (5, 3)])
        k = rng.choice([2, 3, 3, 4]) if w * h >= 12 else 2
        grid, agents = random_instance(rng, w, h, rng.randint(0, 2), k)
        ref = brute_force_soc(grid, agents)
        line = [f"#{i:02d} {w}x{h} k={k} brute={ref}"]
        good = True
        for variant in ("BCP-B", "BCP"):
            r = solve(grid, agents, variant, time_limit=60)
            errs = validate_solution(grid, agents, r["paths"]) if r["paths"] else ["không có lời giải"]
            soc = sum_of_costs(r["paths"]) if r["paths"] else None
            good &= (r["status"] == "optimal" and soc == r["soc"] == ref and not errs)
            line.append(f"{variant}={r['soc']} ({r['status']}, nodes={r.get('nodes', 0)})")
            if errs:
                line.append(f"LỖI: {errs[:2]}")
        ok += good
        print(" | ".join(line), "" if good else "  <-- SAI")
    print(f"\n{ok}/{n_tests} instance khớp với bộ giải vét cạn")
    sys.exit(0 if ok == n_tests else 1)


if __name__ == "__main__":
    main()
