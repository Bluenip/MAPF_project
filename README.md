# Định Tuyến Hai Robot Trong Kho Hàng (2-Agent MAPF)

Dự án cài đặt và đánh giá ba thuật toán giải quyết bài toán định tuyến hai robot trong môi trường kho hàng (Multi-Agent Path Finding - MAPF) bằng **Python Standard Library (thư viện chuẩn, không phụ thuộc bên ngoài)**:

1. **BFS Không gian chung (BFS Joint)**: Tìm kiếm theo chiều rộng trên không gian trạng thái tổ hợp, đảm bảo tối ưu toàn cục makespan.
2. **A* Độc lập + Lập lịch BFS trên tiến trình đường đi cố định (Independent A* + Finite BFS Coordination)**: Tìm đường đi ngắn nhất độc lập bằng A* đơn lẻ, sau đó phối hợp bằng cách chèn bước đợi tối ưu trên không gian chỉ số `(i, j)`. Không đảm bảo tối ưu toàn cục, trả về `None` khi không thể lập lịch.
3. **Joint A* với Heuristic Max Manhattan**: A* trên không gian trạng thái chung với heuristic $h(s) = \max(h_A, h_B)$, đảm bảo tối ưu toàn cục makespan và thường giảm số trạng thái cần xét so với BFS.

---

## 1. Mô Hình Bài Toán & Quy Tắc

### 1.1. Lưới Kho Hàng & Hệ Toạ Độ
- **Bản đồ**: Lưới 2D hình chữ nhật kích thước $R \times C$ gồm các ô có giá trị:
  - `0`: Ô trống hợp lệ để robot di chuyển.
  - `1`: Ô chướng ngại vật / kệ hàng (shelves), robot không được đi vào.
- **Quy ước toạ độ**: Toạ độ dạng tuple `(x, y)` trong đó `x` là chỉ số cột ($0 \le x < C$), `y` là chỉ số hàng ($0 \le y < R$). Trong mảng 2D, ô được truy cập qua `grid[y][x]`.
- **Hành động di chuyển**: Tại mỗi bước thời gian rời rạc $t \in \{0, 1, 2, \dots\}$, mỗi robot thực hiện đồng bộ một trong 5 hành động:
  - Lên: `(0, -1)`
  - Xuống: `(0, 1)`
  - Trái: `(-1, 0)`
  - Phải: `(1, 0)`
  - Đợi tại chỗ (Wait): `(0, 0)`

### 1.2. Quy Tắc Chống Xung Đột (Collision Avoidance)
Hệ thống cấm tuyệt đối hai dạng xung đột sau:
1. **Xung đột đỉnh (Vertex Conflict)**: Hai robot không được cùng chiếm một ô tại bất kỳ thời điểm $t$:
   $$pos_A(t) \ne pos_B(t) \quad \forall t$$
2. **Xung đột hoán đổi cạnh (Edge-Swap Conflict)**: Hai robot không được đồng thời đổi chỗ cho nhau qua cùng một cạnh giữa hai bước $t-1$ và $t$:
   $$\neg \Big( pos_A(t-1) = pos_B(t) \;\land\; pos_B(t-1) = pos_A(t) \Big)$$

### 1.3. Quy Tắc Đích (Goal Rules)
- Robot được phép đến đích cá nhân trước và đứng đợi tại đích.
- Trong quá trình tìm kiếm không gian chung (BFS Joint và Joint A*), robot **được phép rời vị trí đích cá nhân** để tạm thời nhường đường cho robot khác và quay lại đích sau đó nếu điều đó cần thiết để giải quyết xung đột.
- Trạng thái kết thúc hợp lệ là khi cả hai robot **đồng thời ở đích** tại bước thời gian cuối cùng: $pos_A(T) = goal_A$ và $pos_B(T) = goal_B$.

---

## 2. Các Thuật Toán Triển Khai

| Tiêu chí | BFS Joint | A* Độc lập + Lập lịch | Joint A* |
| :--- | :--- | :--- | :--- |
| **Không gian tìm kiếm** | Trạng thái chung: $((x_A, y_A), (x_B, y_B))$ | Pha 1: $2 \times$ Đơn lẻ $(x, y)$<br>Pha 2: Chỉ số $(i, j)$ | Trạng thái chung: $((x_A, y_A), (x_B, y_B))$ |
| **Không gian hành động** | $5 \times 5 = 25$ hành động đồng bộ | Pha 1: 5 hướng<br>Pha 2: $(i+1, j+1), (i+1, j), (i, j+1)$ | $5 \times 5 = 25$ hành động đồng bộ |
| **Hàm Heuristic** | Không dùng ($h = 0$) | Manhattan cho từng robot | $h = \max(\text{Manhattan}_A, \text{Manhattan}_B)$ |
| **Tính tối ưu Makespan** | **Có (Tối ưu toàn cục)** | **Không** | **Có (Tối ưu toàn cục)** |
| **Độ hoàn chỉnh** | Hoàn chỉnh (Complete) | Không hoàn chỉnh (trả về `None` khi bế tắc) | Hoàn chỉnh (Complete) |
| **Khả năng né đường vòng**| Có | Không (chỉ chèn bước đợi trên đường đi cố định) | Có |

### 2.1. BFS Không gian chung (`bfs.py`)
- Tìm kiếm theo chiều rộng từng lớp trên không gian trạng thái $S = V_{free} \times V_{free}$.
- Do chi phí mỗi bước đồng bộ bằng 1, lần đầu tiên trạng thái mục tiêu $(goal_A, goal_B)$ được lấy ra từ hàng đợi `deque` đảm bảo cho ra đường đi có số bước (makespan) nhỏ nhất.
- Điểm hạn chế: Bùng nổ tổ hợp trạng thái ($O(|V|^2)$), tiêu tốn nhiều bộ nhớ và thời gian mở rộng trạng thái khi bản đồ lớn.

### 2.2. A* Độc lập + Lập lịch BFS trên tiến trình cố định (`astar.py`)
Quy trình gồm 2 giai đoạn:
1. **Giai đoạn 1 (Single-Agent A*)**: Chạy thuật toán A* độc lập với heuristic khoảng cách Manhattan cho từng robot để sinh ra hai đường đi ngắn nhất cố định:
   $$Path_A = [p_{A,0}, p_{A,1}, \dots, p_{A,N}], \quad Path_B = [p_{B,0}, p_{B,1}, \dots, p_{B,M}]$$
2. **Giai đoạn 2 (Finite BFS Coordination)**: Tìm đường đi trên lưới chỉ số tiến trình $(i, j)$ với $0 \le i \le N$ và $0 \le j \le M$. Mỗi bước, robot có thể tiến lên một chỉ số hoặc giữ nguyên chỉ số (chèn bước đợi):
   - Chuyển tiếp hợp lệ: $(i+1, j+1)$, $(i+1, j)$, $(i, j+1)$.
   - Loại trừ mọi chuyển tiếp vi phạm xung đột đỉnh hoặc hoán đổi cạnh.
   - Nếu không tồn tại chuỗi chuyển tiếp không va chạm đến $(N, M)$, thuật toán kết luận đường đi cố định không thể lập lịch và trả về `None`.

### 2.3. Joint A* với Heuristic Max Manhattan (`joint_astar.py`)
- Mở rộng A* trên không gian trạng thái chung $((x_A, y_A), (x_B, y_B))$.
- Hàm Heuristic được định nghĩa:
  $$h(s) = \max\Big(|x_A - x_{goalA}| + |y_A - y_{goalA}|, \; |x_B - x_{goalB}| + |y_B - y_{goalB}|\Big)$$
- **Tính chất**: Heuristic này là **chấp nhận được (admissible)** vì thời gian để cả hai robot tới đích không thể nhỏ hơn thời gian của robot ở xa nhất; đồng thời thoả mãn tính **nhất quán (consistent)** do tại mỗi bước di chuyển, khoảng cách Manhattan của mỗi robot giảm tối đa 1, kéo theo $h(s)$ giảm tối đa 1 (bằng đúng chi phí bước đi $c = 1$).
- Do đó, Joint A* luôn tìm được nghiệm tối ưu makespan tương đương BFS nhưng loại bỏ phần lớn các nhánh duyệt không hứa hẹn.

---

## 3. So Sánh: Distance vs Makespan vs CPU Runtime

### 3.1. Định Nghĩa Các Chỉ Số Đo Lường
- **Makespan**: Số bước thời gian đồng bộ từ lúc xuất phát ($t = 0$) cho đến khi cả hai robot đều yên vị tại đích ($t = T$). Makespan $= T = \text{len}(joint\_path) - 1$.
- **Quãng đường di chuyển thực tế (Actual Moves / Path Lengths)**: Số bước mà mỗi robot thực sự đổi ô toạ độ (không tính các bước đợi tại chỗ):
  $$\text{ActualMoves}(A) = \sum_{t=1}^{T} \mathbb{I}(pos_A(t) \ne pos_A(t-1))$$
  Tổng quãng đường thực sự di chuyển = $\text{ActualMoves}(A) + \text{ActualMoves}(B)$.
- **Trạng thái đã mở rộng (Expanded States)**: Số trạng thái được lấy ra xử lý lần đầu, tính cả trạng thái đích (trường hợp đã ở đích từ đầu tính 0). Với A* độc lập, đây là tổng của hai lần tìm đường và bước lập lịch, nên các trạng thái thuộc các không gian khác nhau. Chỉ số này không đo trực tiếp bộ nhớ.
- **Thời gian chạy thuật toán (Runtime)**: Thời gian thực tế đã trôi qua, đo bằng `time.perf_counter()`; có thể thay đổi giữa các lần chạy. Đây không phải số bước thời gian của robot hoặc thời gian CPU riêng của tiến trình.

### 3.2. Bảng Dữ Liệu Thực Nghiệm

> **Ghi chú về khả năng tái lập (Reproducibility):**
> - Mã nguồn **không dùng bất kỳ hàm ngẫu nhiên nào** (không có `random`, không có seed). Thứ tự duyệt hàng xóm cố định và hàng đợi ưu tiên của A* dùng bộ đếm `counter` để phá hoà (tie-breaking) theo thứ tự chèn, nên **Makespan, số bước di chuyển và số trạng thái mở luôn giống hệt nhau** ở mọi lần chạy, trên mọi máy.
> - Riêng **thời gian chạy (ms)** là wall-clock đo bằng `time.perf_counter()`, sẽ dao động theo phần cứng và tải máy; các giá trị dưới đây chỉ mang tính tham khảo. Hãy so sánh dựa trên số trạng thái mở (cố định) thay vì con số ms tuyệt đối.
> - Lệnh tái lập bảng: `python3 demo.py`.

#### Kịch Bản 1: Kho Hàng 10×10 (Start A=(0,0) → Goal A=(5,9); Start B=(4,0) → Goal B=(0,9))
| Thuật toán | Makespan | Di chuyển A | Di chuyển B | Trạng thái mở | Thời gian chạy (ms) | Tính tối ưu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **BFS (Joint)** | **14** | 14 | 13 | 4,792 | ~51.13 ms | Tối ưu toàn cục |
| **A* Độc lập + Lập lịch** | 18 | 14 | 13 | 281 | **~0.27 ms** | Kém tối ưu (+4 bước đợi) |
| **Joint A*** | **14** | 14 | 13 | **266** | **~2.84 ms** | Tối ưu toàn cục |

> **Phân tích chi tiết**:
> - **Joint A* so với BFS**: Cùng đạt makespan tối ưu = 14, nhưng Joint A* chỉ mở **266 trạng thái** so với **4,792 trạng thái** của BFS (giảm khoảng **18 lần**); thời gian chạy theo đó cũng nhanh hơn khoảng **14–18 lần** tùy lần đo.
> - **A* Độc lập**: Mặc dù chạy nhanh nhất (0.27 ms), nhưng do hai đường đi ngắn nhất độc lập giao nhau trong kho hàng, bộ lập lịch phải chèn các bước đợi, khiến makespan tăng lên **18** (kém tối ưu so với 14 của Joint search).

#### Kịch Bản 2: Hành Lang Hẹp Có Vịnh Tránh Xe 3×5 (Passing Bay)
| Thuật toán | Makespan | Di chuyển A | Di chuyển B | Trạng thái mở | Thời gian chạy (ms) | Kết quả |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **BFS (Joint)** | **6** | 6 | 4 | 42 | ~0.26 ms | Thành công (đi vòng qua vịnh) |
| **A* Độc lập + Lập lịch** | **N/A** | N/A | N/A | 20 (5 + 5 + 10) | ~0.04 ms | **Thất bại (Không lập lịch được)** |
| **Joint A*** | **6** | 6 | 4 | **22** | **~0.18 ms** | Thành công (đi vòng qua vịnh) |

---

## 4. Giới Hạn Của Phương Pháp Đường Đi Cố Định (Fixed-Path Failure Limitation)

Phương pháp A* độc lập kết hợp lập lịch trên tiến trình cố định có tốc độ tính toán rất cao nhưng tồn tại **giới hạn bản chất về lý thuyết**:

```
      0   1   2   3   4
 0  [ # ] [ # ] [ . ] [ # ] [ # ]   <-- Ô (2,0) là vịnh tránh xe
 1  [ a ] [ . ] [ . ] [ . ] [ α ]   <-- Hành lang 1 làn duy nhất
 2  [ # ] [ # ] [ . ] [ # ] [ # ]   <-- Ô (2,2) là vịnh tránh xe
```

1. **Nguyên nhân thất bại**:
   - Khi tìm đường độc lập ở Giai đoạn 1, cả Robot A (từ `(0,1)` đến `(4,1)`) và Robot B (từ `(4,1)` đến `(0,1)`) đều chọn đường đi ngắn nhất trực diện dọc theo hàng $y=1$.
   - Ở Giai đoạn 2, không gian phối hợp $(i, j)$ chỉ cho phép **tiến tới** hoặc **dừng đợi** trên chuỗi các ô thuộc đường đi đã chọn:
     - Nếu Robot A đợi ở bất kỳ ô nào trên hàng $y=1$, Robot B vẫn không thể đi qua mặt Robot A vì hành lang chỉ rộng 1 ô (xung đột đỉnh).
     - Nếu hai robot cùng tiến, chúng sẽ đâm vào nhau (xung đột đỉnh) hoặc đổi chỗ cho nhau (xung đột hoán đổi cạnh).
   - Vì không được phép lệch khỏi các ô của $Path_A$ và $Path_B$, thuật toán lập lịch **bất lực** và bắt buộc phải trả về `None`.
2. **Khắc phục bằng Joint Search (BFS / Joint A*)**:
   - Trong không gian trạng thái chung, đồ thị có chứa các trạng thái nơi Robot A tạm thời rẽ lên ô vịnh $(2, 0)$ để nhường đường cho Robot B đi qua ô $(2, 1)$, sau đó Robot A quay trở lại hành lang và tiếp tục về đích.
   - Nhờ đó, BFS Joint và Joint A* dễ dàng giải quyết bài toán với makespan = 6 mà phương pháp đường đi cố định không thể xử lý.

---

## 5. Chỉ Số Từng Giai Đoạn (Stage Metrics)

Khi gọi `astar_independent()`, trường `stage_metrics` cung cấp chi tiết hiệu năng của từng giai đoạn:

```python
{
    "search_a": {
        "expanded": 48,       # Số trạng thái A* đơn lẻ đã mở rộng cho Robot A
        "runtime": 0.000083,  # Thời gian thực thi tìm đường cho Robot A (giây)
        "path_length": 14     # Số bước của đường đi thô của Robot A
    },
    "search_b": {
        "expanded": 38,       # Số trạng thái A* đơn lẻ đã mở rộng cho Robot B
        "runtime": 0.000053,  # Thời gian thực thi tìm đường cho Robot B (giây)
        "path_length": 13     # Số bước của đường đi thô của Robot B
    },
    "coordination": {
        "expanded": 195,      # Số trạng thái (i, j) đã mở rộng trong pha lập lịch
        "runtime": 0.000127   # Thời gian giải quyết xung đột bằng BFS (giây)
    }
}
```

---

## 6. Cấu Trúc Mã Nguồn & API Reference

Toàn bộ mã nguồn nằm trong repo GitHub: https://github.com/Bluenip/MAPF_project

```
code/
├── common.py           # Tiện ích nền tảng: Grid, validation, metrics, định dạng lưới
├── bfs.py              # Cài đặt BFS không gian chung (BFS Joint)
├── astar.py            # Cài đặt A* độc lập & Lập lịch BFS hữu hạn
├── joint_astar.py      # Cài đặt Joint A* với Heuristic Max Manhattan
├── demo.py             # Kịch bản so sánh trực quan và đo đạc thực nghiệm
├── test_algorithms.py  # Bộ 48 bài kiểm thử đơn vị tự động (unittest)
└── README.md           # Tài liệu kỹ thuật chi tiết
```

### Các Hàm Chính
- `common.validate_inputs(grid, start_a, goal_a, start_b, goal_b) -> Grid`: Xác thực tính hợp lệ của lưới (hình chữ nhật, giá trị $\{0, 1\}$), toạ độ nguyên, biên, vật cản, và xung đột vị trí bắt đầu/kết thúc.
- `common.validate_joint_path(grid, path, start_a, goal_a, start_b, goal_b) -> bool`: Kiểm tra kế hoạch đường đi chung đảm bảo hoàn toàn không có xung đột đỉnh hoặc hoán đổi cạnh.
- `bfs.bfs_joint(grid, start_a, goal_a, start_b, goal_b) -> Optional[Dict]`: Chạy BFS không gian chung tìm makespan tối ưu.
- `astar.astar_independent(grid, start_a, goal_a, start_b, goal_b) -> Optional[Dict]`: Chạy A* độc lập rồi phối hợp trên chỉ số $(i, j)$.
- `joint_astar.joint_astar(grid, start_a, goal_a, start_b, goal_b) -> Optional[Dict]`: Chạy Joint A* với heuristic Max Manhattan.

---

## 7. Hướng Dẫn Chạy & Kiểm Thử

Hệ thống được viết hoàn toàn bằng **Python Standard Library** (Python 3.8+, đã kiểm thử trên Python 3.14), không cần `pip install`.

```bash
git clone https://github.com/Bluenip/MAPF_project.git
cd MAPF_project
```

### 7.1. Chạy Toàn Bộ Bộ Kiểm Thử Đơn Vị (48 Tests)
```bash
python3 -m unittest -v
```
Kết quả mong đợi: `48/48 tests OK` (bao gồm kiểm tra tính hợp lệ của lưới, các trường hợp tầm thường, tính tối ưu của BFS & Joint A*, khả năng né tránh, phát hiện xung đột, và kịch bản bế tắc của đường đi cố định).

### 7.2. Chạy Demo Mặc Định
Chạy cả hai kịch bản (Kho hàng 10×10 và Hành lang có vịnh tránh) cùng bảng so sánh chi tiết:
```bash
python3 demo.py
```

### 7.3. Xem Chi Tiết Từng Bước Di Chuyển Trên Bản Đồ (CLI Flag `--steps`)
```bash
# Xem từng bước di chuyển của cả hai kịch bản
python3 demo.py --steps

# Chỉ chạy và xem kịch bản vịnh tránh xe (passing bay)
python3 demo.py --scenario bay --steps

# Chỉ chạy kịch bản kho hàng 10x10
python3 demo.py --scenario warehouse --steps
```

---

## 8. Tóm Tắt & Kết Luận
1. **BFS Joint**: Chuẩn mực về tính tối ưu makespan, đóng vai trò mốc tham chiếu chính xác (ground truth) nhưng chi phí tính toán cao.
2. **Joint A***: Kế thừa đầy đủ tính tối ưu và độ hoàn chỉnh của BFS Joint, nhưng vượt trội về hiệu năng thực thi nhờ hàm heuristic Max Manhattan admissible và consistent.
3. **A* Độc Lập + Lập Lịch**: Rất nhanh cho các bài toán phân tán đơn giản không có đối đầu trực diện, nhưng thiếu tính tối ưu toàn cục và thất bại hoàn toàn trong không gian hẹp đòi hỏi phải đi vòng/lùi xe để nhường đường.
