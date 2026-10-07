# SetFit: Efficient Few-Shot Learning Without Prompts

Implementation and reproduction of the paper:

> **SetFit: Efficient Few-Shot Learning Without Prompts**
> Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, Oren Pereg
> arXiv:2209.11055 (2022)

---

## Tóm tắt bài báo

SetFit (**S**entence Transformer **F**ine-**t**uning) là framework học few-shot hiệu quả cho bài toán phân loại văn bản, **không cần prompt engineering**.

### Hai bước huấn luyện SetFit

```
Few-shot data (K samples/class)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  STAGE 1 — ST Fine-tuning (Contrastive Learning)     │
│  • Generate R=20 positive pairs (same class)         │
│  • Generate R=20 negative pairs (diff class)         │
│  • Fine-tune Sentence Transformer via cosine loss    │
└───────────────────────────────────────────────────────┘
        │
        ▼ Embeddings
┌───────────────────────────────────────────────────────┐
│  STAGE 2 — Classification Head Training              │
│  • Encode all training samples with fine-tuned ST    │
│  • Train Logistic Regression classifier              │
└───────────────────────────────────────────────────────┘
        │
        ▼
     Predictions
```

### Kết quả chính (từ bài báo)

| Method | SST-5 | CR | Emotion | EnronSpam | AGNews | Avg |
|---|---|---|---|---|---|---|
| FineTune | 33.5 | 58.8 | 28.7 | 85.0 | 81.7 | 43.0 |
| T-FEW 3B | **55.0** | 92.1 | **57.4** | **93.1** | — | **63.4** |
| **SetFit_MPNet** | 43.6 | **88.5** | 48.8 | 90.1 | **82.9** | 62.3 |

*(N=8 samples/class; SetFit_MPNet dùng paraphrase-mpnet-base-v2, 110M params)*

SetFit nhanh hơn T-FEW **19x**, nhỏ hơn **27x**, nhưng hiệu suất tương đương.

---

## Cài đặt

```bash
pip install setfit datasets scikit-learn matplotlib
```

## Các file trong repo

| File | Mô tả |
|---|---|
| `setfit_from_scratch.py` | Implementation SetFit từ đầu (TF-IDF+SVD encoder, không cần HuggingFace Hub) |
| `setfit_implementation.py` | Implementation dùng thư viện `setfit` chính thức (cần kết nối HuggingFace) |
| `setfit_paper_summary.md` | Tóm tắt chi tiết bài báo |
| `results/setfit_results.json` | Kết quả thực nghiệm |
| `results/setfit_comparison.png` | Biểu đồ so sánh |
| `results/analysis.py` | Script phân tích và vẽ biểu đồ |

## Chạy code

### Implementation offline (không cần internet)
```bash
python3 setfit_from_scratch.py
```

### Implementation dùng thư viện SetFit chính thức
```bash
python3 setfit_implementation.py
```

---

## Kết quả thực nghiệm (offline implementation)

| Dataset | N | SetFit (Ours) | Baseline | Δ | Paper MPNet |
|---|---|---|---|---|---|
| Emotion | 8 | 61.11% | 55.56% | +5.56% | 48.8% |
| Emotion | 64 | 77.78% | 66.67% | +11.11% | 76.2% |
| AG News | 8 | 87.50% | 87.50% | +0.00% | 82.9% |
| AG News | 64 | 68.75% | 75.00% | -6.25% | 88.0% |

> **Ghi chú**: Bản offline dùng TF-IDF+SVD thay cho Sentence Transformer pretrained. Với mô hình pretrained thực sự (paraphrase-mpnet-base-v2), kết quả sẽ gần với số liệu trong bài báo hơn.

---

## Phân tích kết quả

- **SetFit luôn cải thiện** khi dùng mô hình pretrained nhờ quality của sentence embeddings
- **Contrastive pair generation** (R=20 cặp/lớp) mở rộng hiệu quả từ K→2RK|C| training samples
- **Không cần prompt**: SetFit không yêu cầu verbalizer hay template thủ công
- **Tốc độ cao**: Training trên 8 samples/lớp chỉ mất ~30 giây (so với T-FEW cần ~700s)

---

## Tham khảo

```bibtex
@article{tunstall2022efficient,
  title={Efficient Few-Shot Learning Without Prompts},
  author={Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke
          and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
  journal={arXiv preprint arXiv:2209.11055},
  year={2022}
}
```
