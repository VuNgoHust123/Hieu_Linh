# SetFit: Efficient Few-Shot Learning Without Prompts

**Paper**: Tunstall et al. (2022) — arXiv:2209.11055

## Tóm tắt

SetFit (**S**entence Transformer **F**ine-**t**uning) là phương pháp học few-shot hiệu quả cho bài toán phân loại văn bản, không cần prompt engineering.

## Ý tưởng chính

SetFit gồm **2 bước**:

1. **Fine-tune Sentence Transformer (ST)** theo phương pháp Siamese contrastive learning:
   - Từ K ví dụ có nhãn, tạo R=20 cặp câu positive (cùng lớp) và negative (khác lớp)
   - Fine-tune ST bằng cosine similarity loss
   - Learning rate: 1e-3, batch size: 16, max sequence length: 256, 1 epoch

2. **Huấn luyện classification head**:
   - Dùng ST đã fine-tune để tạo embeddings cho tất cả training samples
   - Huấn luyện Logistic Regression trên các embeddings này

## Các mô hình SetFit

| Biến thể | Backbone | Tham số |
|---|---|---|
| SetFit_RoBERTa | all-roberta-large-v1 | 355M |
| SetFit_MPNet | paraphrase-mpnet-base-v2 | 110M |
| SetFit_MiniLM | paraphrase-MiniLM-L3-v2 | 15M |

## Kết quả chính (N=8 mẫu/lớp)

| Phương pháp | SST-5 | CR | Emotion | EnronSpam | AGNews | Avg |
|---|---|---|---|---|---|---|
| FineTune | 33.5 | 58.8 | 28.7 | 85.0 | 81.7 | 43.0 |
| T-FEW 3B | **55.0** | 92.1 | **57.4** | **93.1** | — | **63.4** |
| SetFit_MPNet | 43.6 | **88.5** | 48.8 | 90.1 | **82.9** | 62.3 |

SetFit_MPNet đạt kết quả tương đương T-FEW (3B params) trong khi chỉ có 110M params và nhanh hơn **19x** về tốc độ training.

## Ưu điểm

- Không cần prompt / verbalizer
- Không cần mô hình ngôn ngữ khổng lồ (1B+ params)
- Nhanh hơn 19-123x so với T-FEW
- Áp dụng được cho đa ngôn ngữ
- Hiệu quả trong few-shot distillation
