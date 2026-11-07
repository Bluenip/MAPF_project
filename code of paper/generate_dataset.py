"""Sinh bộ dữ liệu MÔ PHỎNG theo thiết lập thí nghiệm của Lam et al. (IJCAI 2019), mục 5.1.

LƯU Ý: đây KHÔNG phải bộ dữ liệu của paper (paper không công bố file). Chỉ giống paper ở
bản đồ, số agent và số instance; chỉ 50 instance (lak503d, 100 agent) là bản gốc.

Paper dùng "cùng bộ instance với [Li et al., AAAI 2019]": 1.350 instance trên 4 bản đồ,
50 instance cho mỗi số agent. Tác giả không công bố trọn bộ, nên script này:

  * lak503d, 100 agent : dùng nguyên 50 instance GỐC của Li et al. (tải từ repo CBSH2).
  * các cấu hình còn lại: sinh ngẫu nhiên theo cùng cách (start/goal ngẫu nhiên, khác nhau
    từng đôi, nằm trong cùng thành phần liên thông), với seed cố định để tái lập được.

Số agent đọc từ trục hoành Hình 3 của paper (5 + 4 + 10 + 8 = 27 cấu hình x 50 = 1.350):
  20x20        : 20, 30, 40, 50, 60
  10obs-20x20  : 20, 30, 40, 50
  den520d      : 30, 40, ..., 120
  lak503d      : 30, 40, ..., 100

Chạy:  .venv/bin/python generate_dataset.py
"""
import glob
import os
import random

from mapf_io import Grid, load_map, save_map, save_scen

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "dataset")
MAPS = os.path.join(DATA, "maps")
SCEN = os.path.join(DATA, "scenarios")
ORIG = os.path.join(DATA, "original_li2019", "lak503d")
N_INSTANCES = 50

AGENT_COUNTS = {
    "20x20": [20, 30, 40, 50, 60],
    "10obs-20x20": [20, 30, 40, 50],
    "den520d": list(range(30, 121, 10)),
    "lak503d": list(range(30, 101, 10)),
}


def make_small_maps():
    save_map(os.path.join(MAPS, "20x20.map"), ["." * 20] * 20)
    # 10% ô trống được thay bằng chướng ngại vật (40 ô), giữ bản đồ liên thông.
    # Seed cố định 2019: vị trí 40 ô chướng ngại luôn giống nhau => bản đồ 10obs-20x20 tái lập được.
    rng = random.Random(2019)
    while True:
        cells = [(x, y) for y in range(20) for x in range(20)]
        obs = set(rng.sample(cells, 40))
        rows = ["".join("@" if (x, y) in obs else "." for x in range(20)) for y in range(20)]
        g = Grid(rows)
        if len(g.largest_component()) == 400 - 40:
            save_map(os.path.join(MAPS, "10obs-20x20.map"), rows)
            return


def convert_lak503d():
    """Bản đồ của Li et al. có định dạng "hàng,cột" + lưới; đổi sang định dạng MovingAI."""
    with open(os.path.join(ORIG, "lak503d.map")) as f:
        lines = f.read().split("\n")
    r, c = map(int, lines[0].split(","))
    save_map(os.path.join(MAPS, "lak503d.map"), [l[:c] for l in lines[1 : 1 + r]])


def original_lak503d_instances():
    """Đọc 50 instance gốc (mỗi dòng: start_row,start_col,goal_row,goal_col)."""
    out = {}
    for f in glob.glob(os.path.join(ORIG, "lak503dmap-100agents-*.agents")):
        idx = int(f.rsplit("-", 1)[1].split(".")[0])
        lines = open(f).read().split("\n")
        agents = []
        for line in lines[1 : 1 + int(lines[0])]:
            sr, sc, gr, gc = map(int, line.strip().strip(",").split(","))
            agents.append(((sc, sr), (gc, gr)))
        out[idx] = agents
    return out


def random_instance(cells, k, rng):
    starts = rng.sample(cells, k)
    goals = rng.sample(cells, k)
    return list(zip(starts, goals))


def main():
    os.makedirs(MAPS, exist_ok=True)
    make_small_maps()
    convert_lak503d()
    orig = original_lak503d_instances()
    total = 0
    for name, counts in AGENT_COUNTS.items():
        grid = load_map(os.path.join(MAPS, name + ".map"))
        cells = sorted(grid.largest_component())
        out_dir = os.path.join(SCEN, name)
        os.makedirs(out_dir, exist_ok=True)
        for k in counts:
            # Seed cố định theo (bản đồ, số agent), ví dụ "den520d-60": chạy lại luôn sinh ra đúng các file
            # .scen như trong dataset/ (seed dạng chuỗi được Python băm bằng SHA-512, không phụ thuộc máy).
            rng = random.Random(f"{name}-{k}")
            for i in range(N_INSTANCES):
                if name == "lak503d" and k == 100 and i in orig:
                    agents = orig[i]
                else:
                    agents = random_instance(cells, k, rng)
                save_scen(os.path.join(out_dir, f"{name}-{k}agents-{i}.scen"), name + ".map", grid, agents)
                total += 1
        print(f"{name:12s}: {len(counts)} số agent x {N_INSTANCES} = {len(counts) * N_INSTANCES} instance")
    print(f"Tổng: {total} instance -> {SCEN}")


if __name__ == "__main__":
    main()
