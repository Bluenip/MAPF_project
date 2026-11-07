# Cài đặt lại: Branch-and-Cut-and-Price cho MAPF (Lam et al., IJCAI 2019)

Cài đặt lại bằng Python bài báo

> E. Lam, P. Le Bodic, D. Harabor, P. J. Stuckey. *Branch-and-Cut-and-Price for Multi-Agent Pathfinding*. IJCAI 2019, tr. 1289–1296.

kèm một bộ dữ liệu **mô phỏng theo thiết lập thí nghiệm** của paper (mục 5.1). Paper không công bố file dataset, nên bộ này **không giống** bộ của paper (xem mục "Bộ dữ liệu" bên dưới).

**Mọi số liệu về BCP trên slide (slide 28, 30, 31 và bảng ở slide 37) là kết quả công bố trong paper**, không phải do code này chạy ra. Ngoại lệ duy nhất: kết quả `60/60` (slide 35, 37) do `test_bcp.py` chạy ra.
Code chính thức của tác giả (C++, SCIP + CPLEX/Gurobi): https://github.com/ed-lam/bcp-mapf

## Cấu trúc thư mục

| File / thư mục | Nội dung |
|---|---|
| `bcp_mapf.py` | Bộ giải BCP: master problem, pricer A*, các separator, rẽ nhánh, branch-and-bound |
| `mapf_io.py` | Đọc/ghi `.map`, `.scen` (định dạng MovingAI), kiểm tra lời giải (validator) |
| `generate_dataset.py` | Sinh bộ 1.350 instance mô phỏng theo thiết lập của paper (cùng bản đồ, số agent, số instance; chỉ 50 instance là bản gốc) |
| `run_experiments.py` | Chạy thí nghiệm, ghi kết quả ra CSV |
| `test_bcp.py` | Kiểm thử: so SOC tối ưu của BCP/BCP-B với bộ giải vét cạn |
| `dataset/maps/` | 4 bản đồ: `20x20`, `10obs-20x20`, `den520d`, `lak503d` |
| `dataset/scenarios/<map>/` | `<map>-<k>agents-<i>.scen`, 50 instance cho mỗi số agent k |
| `dataset/original_li2019/` | File gốc tải từ repo CBSH2 của Li et al. (lak503d, 100 agent) |
| `requirements.txt` | Phiên bản thư viện đã dùng (`highspy`, `numpy`, `matplotlib`) |

## Tái lập kết quả trên slide (slide 35, 37)

Slide chỉ báo cáo **một** kết quả của phần này: *BCP và BCP-B khớp với bộ giải vét cạn trên 60 instance nhỏ*.
Các bảng số liệu về BCP trên slide 28 và 30 là **kết quả công bố trong paper**, không phải do code này chạy ra.

```bash
cd "code of paper"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # highspy 1.15.1, numpy 2.5.4, matplotlib 3.11.2 (đã thử với Python 3.14.9)
.venv/bin/python test_bcp.py 60
```

Kết quả mong đợi ở dòng cuối: `60/60 instance khớp với bộ giải vét cạn` (mất khoảng vài chục giây).

**Về tính ngẫu nhiên.** Mọi chỗ dùng `random` đều có seed cố định và có chú thích ngay trong code:
- `test_bcp.py`: `random.Random(7)` sinh 60 instance kiểm thử, lần chạy nào cũng giống nhau.
- `bcp_mapf.py`: `random.Random(0)` xáo thứ tự agent cho Prioritized Planning (chỉ để tìm cận trên ban đầu; không ảnh hưởng SOC tối ưu).
- `generate_dataset.py`: `random.Random(2019)` cho bản đồ `10obs-20x20` và `random.Random("<map>-<k>")` cho từng cấu hình. Chạy lại script sẽ sinh ra đúng thư mục `dataset/` **đang có trong repo này**, giống hệt từng byte (không phải dataset của paper, vì paper không công bố).

Repo không kèm kết quả benchmark nào chạy bằng code này: slide không dùng các kết quả đó, và chúng không so sánh trực tiếp được với paper vì dataset khác nhau. Nếu chạy `run_experiments.py`, kết quả sẽ được ghi vào `results/` (đã có trong `.gitignore`). Thời gian chạy phụ thuộc máy.

## Cài đặt và chạy (đầy đủ)

```bash
cd "code of paper"
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python test_bcp.py 60                 # kiểm thử tính đúng đắn
.venv/bin/python generate_dataset.py            # (tuỳ chọn) sinh lại dataset của repo — giống hệt bản trong repo vì seed cố định

# chạy thử nhanh
.venv/bin/python run_experiments.py --maps 20x20 --agents 20 30 --instances 5 --time-limit 60
# toàn bộ 1.350 instance x 2 biến thể, timeout 5 phút như paper (dataset mô phỏng, không phải bộ của paper) — mất rất nhiều giờ
.venv/bin/python run_experiments.py --time-limit 300
```

Kết quả được ghi vào `results/results.csv`, mỗi instance một dòng. Script tự bỏ qua các instance đã chạy, nên có thể dừng giữa chừng rồi chạy tiếp.
Các cột chính: `status` (optimal/timeout), `soc`, `lb` (cận dưới), `root_lb` (cận dưới tại nút gốc), `time`, `nodes`, số lát cắt từng loại, và `valid` (lời giải đã qua validator).

## Ánh xạ paper → code

| Paper | Code (`bcp_mapf.py`) |
|---|---|
| Master problem (1)–(3) | `BCPSolver.__init__`, `add_column` (LP giải bằng HiGHS) |
| Xung đột đỉnh (5) | `separate` → `add_vertex_row` (tính cả việc agent đứng ở đích sau khi tới) |
| Xung đột cạnh (6) | `separate` → `add_edge_row` |
| Pricer, hàm mục tiêu (7) | `price`: A* trên đồ thị mở rộng theo thời gian, phạt −β, −γ, −π, −ρ |
| Rẽ nhánh agent–đỉnh (3.6) | `branch` (phần sau) |
| Corridor conflict (8) — chỉ BCP | `separate_corridor` |
| Rectangle symmetry (9) — chỉ BCP | `separate_rectangle`, `_rectangle` |
| Rẽ nhánh theo độ dài (4.3) — chỉ BCP | `branch` (phần đầu) |

## Bộ dữ liệu: cái gì giống paper, cái gì tự sinh

Paper chỉ ghi là dùng "cùng bộ instance với [Li et al., AAAI 2019]", gồm 1.350 instance trên 4 bản đồ, 50 instance cho mỗi số agent, và **không công bố file**. Repo của tác giả chỉ có bộ benchmark MovingAI dùng cho bản journal năm 2022, không phải bộ của năm 2019.

| Bản đồ | Số agent (đọc từ Hình 3) | Nguồn |
|---|---|---|
| `20x20` (lưới trống) | 20, 30, 40, 50, 60 | Tự sinh |
| `10obs-20x20` (10% chướng ngại) | 20, 30, 40, 50 | Tự sinh, 1 bản đồ cố định (seed 2019) |
| `den520d` | 30, 40, …, 120 | Bản đồ từ MovingAI (lấy qua repo `bcp-mapf`), instance tự sinh |
| `lak503d` | 30, 40, …, 100 | Bản đồ gốc của Li et al.; **k = 100 dùng đúng 50 instance gốc**, các k khác tự sinh |

5 + 4 + 10 + 8 = 27 cấu hình × 50 = **1.350 instance**, đúng bằng **số lượng** trong paper. Chỉ 50 instance (lak503d, k = 100) là bản gốc; 1.300 instance còn lại là tự sinh: start và goal ngẫu nhiên, đôi một khác nhau, nằm trong thành phần liên thông lớn nhất, với seed cố định. Vì vậy **kết quả chạy trên bộ dữ liệu này không so sánh trực tiếp được với Hình 3 và Hình 4 của paper**.

## Khác biệt so với cài đặt gốc (cần ghi vào báo cáo)

1. **Ngôn ngữ và bộ giải.** Bản này viết bằng Python, dùng HiGHS. Bản gốc viết bằng C++, dùng SCIP + CPLEX. Bản Python chậm hơn nhiều. Code này chỉ được kiểm chứng về **tính đúng đắn** (khớp bộ giải vét cạn, `test_bcp.py`), chưa được dùng để tái hiện số liệu thí nghiệm của paper.
2. **Không có các tính năng sẵn của SCIP**: heuristic tìm nghiệm, chọn nút kiểu diving, quản lý pool lát cắt. Thay vào đó code dùng Prioritized Planning để có nghiệm cận trên ban đầu, và chọn nút theo best-first.
3. **Horizon.** Để thuật toán luôn dừng ở các nút nhánh vô nghiệm, độ dài đường đi được chặn bởi T = max_a(UB − Σ_{b≠a} dist_b). Mọi nghiệm tối ưu đều thoả giới hạn này, nên kết quả vẫn tối ưu chính xác (xem docstring của `solve`).
4. **Rectangle cut (9).** Paper dựa vào chứng minh của Li et al. Ở đây dùng một cách xây dựng có chứng minh riêng (ghi trong docstring): hai agent cùng đi "không trễ" theo một hướng chéo, một agent cắt ngang và một agent cắt dọc qua hình chữ nhật, thì bắt buộc gặp nhau tại cùng một ô và cùng một thời điểm.
5. **Chi phí.** Code dùng SOC chuẩn (thời điểm đến đích lần cuối). Paper tính "số vị trí" = SOC + 1 cho mỗi agent. Hai cách chỉ lệch nhau một hằng số, nên nghiệm tối ưu là như nhau.
