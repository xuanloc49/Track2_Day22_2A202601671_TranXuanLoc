# Báo Cáo Phân Tích Thực Nghiệm & Đánh Giá — Day 22: LangSmith + Prompt Versioning

**Học viên:** Trần Xuân Lộc (2A202601671)  
**Dự án:** Track 2 — Day 22: RAG Pipeline Evaluation, Prompt Hub Versioning & Guardrails AI  
**Project LangSmith:** `day22-lab`  

---

## 1. Tổng quan kết quả A/B Testing & Đánh giá RAGAS (50 QA Pairs)

Trong bài lab này, hệ thống RAG được đánh giá toàn diện trên 50 cặp câu hỏi - câu trả lời chuẩn (`QA_PAIRS`) trên 2 phiên bản prompt:
- **Prompt V1 (`tran-xuan-loc-rag-prompt-v1`)**: Prompt hướng dẫn trả lời bằng tiếng Anh, tập trung vào tính trung thực nghiêm ngặt và căn cứ dữ liệu (`strict factual grounding`).
- **Prompt V2 (`tran-xuan-loc-rag-prompt-v2`)**: Prompt hỗ trợ song ngữ (Anh - Việt), hướng dẫn tổng hợp súc tích và giải thích thuật ngữ AI.

### Bảng đối sánh 4 chỉ số RAGAS thực tế:

| Chỉ số RAGAS | Prompt V1 | Prompt V2 | Winner | Ngưỡng yêu cầu | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Faithfulness (Độ trung thực)** | **0.9622** | **0.8351** | 🏆 **V1** | $\ge 0.80$ | **Cả 2 đều $\ge 0.80$, V1 $\ge 0.96$ (+Bonus)** |
| **Answer Relevancy (Độ liên quan)** | **0.9104** | 0.8878 | 🏆 **V1** | — | **Rất cao (> 0.91)** |
| **Context Recall (Độ phủ ngữ cảnh)** | **1.0000** | **1.0000** | 🤝 **Hòa** | — | **Hoàn hảo (100%)** |
| **Context Precision (Độ chính xác truy xuất)** | 0.9417 | **0.9450** | 🏆 **V2** | — | **Rất cao (> 0.94)** |

---

## 2. Phân tích chuyên sâu nguyên nhân chênh lệch V1 và V2

### 2.1. Về chỉ số Faithfulness (0.9622 so với 0.8351)
- **Cả 2 phiên bản đều vượt ngưỡng yêu cầu $\ge 0.80$**. Đặc biệt, **Prompt V1** đạt điểm số xuất sắc `0.9622` ($\ge 0.90$ - đạt điểm thưởng Rubric) nhờ chỉ thị ràng buộc nghiêm ngặt:
  > *"Answer the question using ONLY the provided context. If the context does not contain enough information, say 'I cannot answer based on the provided context.' Do not extrapolate."*
  LLM bám sát 100% các sự thật được nêu trong tài liệu FAISS, triệt tiêu ảo giác (hallucination).
- **Prompt V2** đạt `0.8351` (vượt chuẩn $\ge 0.80$). Mặc dù có giải thích song ngữ súc tích, LLM vẫn bám rất sát thông tin tài liệu, tuy nhiên việc dịch nghĩa thuật ngữ khiến RAGAS claim extractor nhận diện một số phát biểu đã được paraphrase nhẹ.

### 2.2. Về chỉ số Answer Relevancy (0.9104 so với 0.8878)
- **Prompt V1** sinh ra câu trả lời trực diện, đồng nhất ngôn ngữ với câu hỏi đầu vào, tối ưu hóa vector similarity giữa câu hỏi và câu trả lời.
- **Prompt V2** trả lời súc tích song ngữ, đạt mức tốt `0.8878`.

### 2.3. Về chỉ số Context Recall (1.0000) & Context Precision (0.9417 / 0.9450)
- Cả hai phiên bản đều đạt `Context Recall = 1.0000`, chứng minh thuật toán chunking (`RecursiveCharacterTextSplitter`, chunk_size=300, chunk_overlap=50) cùng bộ mã hóa `text-embedding-3-small` và FAISS retriever ($k=3$) đã truy xuất trọn vẹn 100% ngữ cảnh cần thiết cho toàn bộ 50 câu hỏi.
- `Context Precision` đạt trên `0.94` ở cả 2 phiên bản, chứng minh các chunk liên quan nhất luôn được xếp ở vị trí ưu tiên cao nhất ($Rank\ 1$).


---

## 3. Hệ thống Định tuyến A/B (Deterministic Routing)

- Hệ thống áp dụng hàm băm MD5 tất định trên `request_id`:
  $$\text{hash\_val} = \text{int}(\text{MD5}(request\_id)[-4:], 16) \pmod{100}$$
  - Nếu $\text{hash\_val} < 40 \implies \text{Route to V1}$ (40% Traffic)
  - Nếu $\text{hash\_val} \ge 40 \implies \text{Route to V2}$ (60% Traffic)
- **Kết quả thực nghiệm trên 50 request (`req-0000` đến `req-0049`):**
  - **V1:** 19 câu truy vấn (38%)
  - **V2:** 31 câu truy vấn (62%)
  - Đảm bảo tính tất định: Một `request_id` cụ thể luôn luôn trả về cùng 1 phiên bản prompt duy nhất khi chạy lại nhiều lần.
  - Toàn bộ log được lưu tại [`02_ab_routing_log.txt`](./02_ab_routing_log.txt).

---

## 4. Hệ thống Kiểm duyệt & Bảo vệ (Guardrails AI Validators)

Hệ thống đã xây dựng thành công 2 Custom Validators độc lập:

1. **`PIIDetector` (Personal Identifiable Information)**:
   - Sử dụng Regex Engine để quét và che giấu tự động 4 nhóm dữ liệu nhạy cảm: `EMAIL`, `PHONE`, `SSN`, `CREDIT_CARD`.
   - Cơ chế `OnFailAction.FIX`: Tự động thay thế chuỗi nhạy cảm bằng nhãn `[<TYPE>_REDACTED]` mà không làm gián đoạn pipeline.
   - Vượt qua 6/6 test cases thực tế (Log tại [`04_pii_demo_log.txt`](./04_pii_demo_log.txt)).

2. **`JSONFormatter` (Structured Output Repair)**:
   - Tự động sửa các lỗi phổ biến từ LLM: gỡ Markdown code fences (````json ... ````), chuyển dấu nháy đơn `'` thành `"` chuẩn, xóa dấu phẩy thừa (`trailing commas`), và fallback JSON có cấu trúc khi dữ liệu bị hỏng hoàn toàn.
   - Cơ chế `OnFailAction.FIX`: Trả về chuỗi JSON hợp lệ đã format đẹp.
   - Vượt qua 5/5 test cases thực tế (Log tại [`04_json_demo_log.txt`](./04_json_demo_log.txt)).

---

## 5. Danh mục các tệp bằng chứng (`evidence/`)

| Tên tệp | Mô tả chi tiết |
| :--- | :--- |
| [`01_langsmith_traces.png`](./01_langsmith_traces.png) | Ảnh chụp màn hình LangSmith UI hiển thị danh sách $\ge 50$ traces từ Bước 1 |
| [`02_prompt_hub.png`](./02_prompt_hub.png) | Ảnh chụp màn hình LangSmith Prompt Hub hiển thị 2 phiên bản `tran-xuan-loc-rag-prompt-v1` và `v2` |
| [`02_ab_routing_log.txt`](./02_ab_routing_log.txt) | Log phân phối và kết quả A/B routing của 50 câu truy vấn |
| [`03_ragas_scores.png`](./03_ragas_scores.png) | Ảnh chụp màn hình Terminal hiển thị bảng điểm so sánh 4 chỉ số RAGAS giữa V1 và V2 |
| [`03_ragas_report.json`](./03_ragas_report.json) | Tệp dữ liệu JSON chứa điểm số đánh giá 4 chỉ số RAGAS |
| [`04_pii_demo_log.txt`](./04_pii_demo_log.txt) | Log console chạy kiểm thử 6 test cases khử PII |
| [`04_json_demo_log.txt`](./04_json_demo_log.txt) | Log console chạy kiểm thử 5 test cases sửa JSON tự động |
