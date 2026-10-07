# Hướng dẫn chạy SetFit trên Google Colab

> Dành cho người mới bắt đầu — không cần cài đặt gì trên máy tính cá nhân!

---

## Google Colab là gì?

Google Colab là một công cụ **miễn phí** của Google cho phép bạn chạy code Python trực tiếp trên trình duyệt web, sử dụng máy chủ của Google (có GPU miễn phí).

---

## Bước 1 — Mở Google Colab

1. Truy cập: **https://colab.research.google.com**
2. Đăng nhập bằng tài khoản Google của bạn
3. Nhấn **"New notebook"** (Tạo notebook mới)

---

## Bước 2 — Cài đặt thư viện

Trong ô code đầu tiên (cell), dán đoạn sau rồi nhấn **Shift + Enter** để chạy:

```python
# Cài đặt các thư viện cần thiết
!pip install setfit datasets scikit-learn matplotlib
```

> ⏳ Chờ khoảng 1-2 phút để cài xong. Bạn sẽ thấy dòng chữ xanh khi hoàn thành.

---

## Bước 3 — Chạy code SetFit (dùng thư viện chính thức)

Tạo ô code mới (nhấn **+ Code**), dán toàn bộ đoạn code bên dưới, rồi nhấn **Shift + Enter**:

```python
"""
SetFit: Efficient Few-Shot Learning Without Prompts
Paper: Tunstall et al. (2022) - arXiv:2209.11055

Thử nghiệm SetFit trên dataset Emotion (phân loại cảm xúc 6 lớp)
với N=8 mẫu mỗi lớp (few-shot learning).
"""

# Tắt wandb (tránh bị hỏi tài khoản khi training)
import os
os.environ["WANDB_DISABLED"] = "true"

import random
from collections import defaultdict
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

# ── Cấu hình ──────────────────────────────────────────────
MODEL_NAME  = "sentence-transformers/paraphrase-mpnet-base-v2"  # 110M params
N_PER_CLASS = 8    # số mẫu mỗi lớp — thử thay thành 16, 32, 64
NUM_ITERS   = 20   # số cặp câu (R) — giữ nguyên theo bài báo
SEED        = 42

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# ── Hàm lấy mẫu few-shot ──────────────────────────────────
def sample_few_shot(dataset, label_col, n, seed=42):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, ex in enumerate(dataset):
        by_class[ex[label_col]].append(i)
    indices = []
    for label, idxs in by_class.items():
        indices.extend(rng.sample(idxs, min(n, len(idxs))))
    return dataset.select(indices)

# ── 1. Tải dataset ─────────────────────────────────────────
print("📥 Đang tải dataset Emotion...")
raw      = load_dataset("dair-ai/emotion")
train_ds = raw["train"]
test_ds  = raw["test"]

few_shot = sample_few_shot(train_ds, "label", N_PER_CLASS, SEED)
print(f"   Train few-shot: {len(few_shot)} mẫu ({N_PER_CLASS} mẫu/lớp)")
print(f"   Test:           {len(test_ds)} mẫu\n")

# ── 2. Tải mô hình SetFit ──────────────────────────────────
print("🤖 Đang tải mô hình Sentence Transformer...")
model = SetFitModel.from_pretrained(
    MODEL_NAME,
    labels=list(range(6)),
)

# ── 3. Huấn luyện (2 giai đoạn) ───────────────────────────
print("\n🏋️ Bắt đầu huấn luyện SetFit (2 giai đoạn)...")
print("   Giai đoạn 1: Fine-tune Sentence Transformer (contrastive learning)")
print("   Giai đoạn 2: Train Logistic Regression classifier\n")

args = TrainingArguments(
    batch_size     = 16,
    num_epochs     = 1,
    num_iterations = NUM_ITERS,
    seed           = SEED,
)

trainer = Trainer(
    model          = model,
    args           = args,
    train_dataset  = few_shot,
    column_mapping = {"text": "text", "label": "label"},
)

trainer.train()
print("✅ Huấn luyện xong!\n")

# ── 4. Đánh giá ────────────────────────────────────────────
print("📊 Đánh giá trên tập test...")
preds  = model.predict(test_ds["text"])
labels = test_ds["label"]

acc = accuracy_score(labels, preds)
print(f"\n🎯 Accuracy: {acc*100:.2f}%")
print(f"   (Kết quả trong bài báo với N=8: 48.8%)\n")

print("Chi tiết theo từng lớp cảm xúc:")
print(classification_report(labels, preds, target_names=LABEL_NAMES))
```

---

## Bước 4 — Vẽ biểu đồ kết quả

Tạo ô code mới, dán code bên dưới và chạy:

```python
import matplotlib.pyplot as plt

# Kết quả từ bài báo (SetFit_MPNet) và FineTune baseline
n_values    = [8, 16, 32, 64]
paper_setfit = [48.8, 58.3, 66.4, 76.2]   # từ bài báo
paper_base   = [28.7, 35.1, 42.0, 65.0]   # FineTune baseline (ước tính)

plt.figure(figsize=(8, 5))
plt.plot(n_values, paper_setfit, "o-", color="#2196F3", linewidth=2.5,
         markersize=8, label="SetFit_MPNet (bài báo)")
plt.plot(n_values, paper_base, "s--", color="#FF7043", linewidth=2,
         markersize=7, label="FineTune baseline")

# Đánh dấu điểm bạn vừa chạy (N=8)
plt.scatter([8], [acc * 100], color="#4CAF50", s=150, zorder=5,
            label=f"Kết quả của bạn (N=8): {acc*100:.1f}%")

plt.xlabel("Số mẫu huấn luyện mỗi lớp (N)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("SetFit — Emotion Classification\n(Few-shot Learning)", fontsize=13, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.xticks(n_values)
plt.tight_layout()
plt.savefig("setfit_emotion_result.png", dpi=150)
plt.show()
print("✅ Biểu đồ đã lưu!")
```

---

## Bước 5 — Thử nghiệm thêm (tùy chọn)

Bạn có thể thay đổi các thông số ở **Bước 3** để xem kết quả thay đổi:

| Thay đổi | Giá trị thử | Kỳ vọng |
|---|---|---|
| `N_PER_CLASS` | 16, 32, 64 | Accuracy tăng theo N |
| `MODEL_NAME` | `"sentence-transformers/paraphrase-MiniLM-L3-v2"` | Nhỏ hơn, nhanh hơn |
| Dataset | `"ag_news"` | Topic classification (4 lớp) |

---

## Bước 6 — Thử với dataset AG News (phân loại tin tức)

```python
# Thay dataset Emotion bằng AG News
raw_ag   = load_dataset("fancyzhx/ag_news")
train_ag = raw_ag["train"]
test_ag  = raw_ag["test"]

few_shot_ag = sample_few_shot(train_ag, "label", N_PER_CLASS, SEED)
print(f"AG News — Train: {len(few_shot_ag)} mẫu, Test: {len(test_ag)} mẫu")

model_ag = SetFitModel.from_pretrained(MODEL_NAME, labels=list(range(4)))
trainer_ag = Trainer(
    model          = model_ag,
    args           = args,
    train_dataset  = few_shot_ag,
    column_mapping = {"text": "text", "label": "label"},
)
trainer_ag.train()

preds_ag  = model_ag.predict(test_ag["text"])
labels_ag = test_ag["label"]
acc_ag    = accuracy_score(labels_ag, preds_ag)
print(f"\n🎯 AG News Accuracy: {acc_ag*100:.2f}%")
print(f"   (Kết quả trong bài báo với N=8: 82.9%)")
```

---

## Câu hỏi thường gặp

**Q: Xuất hiện dòng chữ "wandb: Enter your choice" là sao?**
> Đó là công cụ theo dõi thí nghiệm Weights & Biases hỏi bạn có muốn đăng nhập không. Gõ **`3`** rồi Enter để bỏ qua. Hoặc thêm `os.environ["WANDB_DISABLED"] = "true"` vào đầu cell là sẽ không bị hỏi nữa (đã có sẵn trong code bên trên).

**Q: Chạy mất bao lâu?**
> Khoảng 3–5 phút với GPU miễn phí của Colab (T4). Nếu dùng CPU thì lâu hơn (~15 phút).

**Q: Làm sao bật GPU?**
> Vào menu **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**.

**Q: Kết quả của tôi thấp hơn bài báo một chút, có bình thường không?**
> Hoàn toàn bình thường! Bài báo lấy trung bình 10 lần chạy khác nhau (10 random seeds) rồi báo cáo kết quả trung bình. Bạn chỉ chạy 1 lần nên có thể lệch ±3–5%.

**Q: Lưu notebook ở đâu?**
> Notebook tự động lưu vào Google Drive của bạn tại thư mục **"Colab Notebooks"**.

---

## Tóm tắt nhanh

```
1. Mở colab.research.google.com
2. New notebook
3. Cài thư viện:  !pip install setfit datasets scikit-learn matplotlib
4. Dán code Bước 3 → Shift+Enter
5. Dán code Bước 4 → Shift+Enter (xem biểu đồ)
```

Chúc bạn thành công! 🎉
