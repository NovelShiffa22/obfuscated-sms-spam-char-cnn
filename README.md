# Obfuscated SMS Spam Detection with a Character-Level CNN

Code, trained model, and evaluation artifacts for the paper **"Detection of Obfuscated SMS Spam Using Character-Level Convolutional Neural Networks"**.

The project detects Indonesian SMS spam that uses character-level obfuscation (leetspeak digits, inserted dots, dropped vowels). A Char-CNN with parallel 1D convolutions reads raw characters, so it does not depend on a word vocabulary that obfuscation breaks.

## Repository contents

| File | Description |
|---|---|
| `01_Proposed_Model_Training.ipynb` | Corpus audit, template grouping, grouped train/val/test split, training augmentation, Char-CNN training, holdout test, augmentation ablation |
| `02_Model_Comparison_Evaluation.ipynb` | 10-model benchmark, adversarial robustness suite, multi-seed stability, group-cluster bootstrap CIs, efficiency measurements |
| `models/char_cnn_spam_model.keras` | Trained Char-CNN (restored best-validation-loss checkpoint) |
| `models/tokenizer_char.pkl` | Deterministic character tokenizer (41-character vocabulary) |
| `results/split_manifest.csv` | Privacy-safe split manifest: `msg_id`, `group_id`, `source_group`, `label`, `partition`, `outer_fold` |
| `results/epoch_validation_log.csv` | Per-epoch training and validation loss/accuracy |
| `results/test_predictions_manifest.csv` | Per-message test probabilities and predictions for all 10 models |
| `results/benchmark_holdout_results.csv` | Holdout test metrics for all models |
| `results/ablation_study_results.csv` | Augmentation ablation (Char-CNN vs. Word-CNN) |
| `requirements.txt` | Exact library versions |

## Method summary

- **Corpus:** 392 labeled Indonesian SMS (317 spam, 75 ham) after dropping 36 unlabeled rows; 253 unique cleaned texts.
- **Preprocessing:** lowercase, keep a 41-character alphabet (`a-z`, `0-9`, space, `. , ? !`), pad/truncate to 300 characters.
- **Leakage control:** near-duplicate messages (character 3-5 gram TF-IDF cosine similarity >= 0.80, same label) are merged into template groups; `StratifiedGroupKFold` splits by group. Cross-partition group overlap is 0.
- **Split (unique texts):** 151 train / 51 validation / 51 test.
- **Training augmentation (train split only):** each spam template becomes 5 rows (2 clean + leetspeak + dot insertion + vowel removal); ham is oversampled to balance, with numeric/colloquial suffixes (e.g. "jam 10", "no 482", "ya", "bro") on about 40% of the oversampled copies. Result: 1,210 rows (605 spam / 605 ham).
- **Model:** Embedding (50-d, L2) with a frozen padding mask, SpatialDropout (0.2), three parallel Conv1D branches (kernel 3/4/5, 64 filters each) with global max pooling, Dense-128, Dropout 0.5, sigmoid output. Adam (lr 3e-4), batch 32, max 35 epochs, early stopping on validation loss (patience 5), seed 42.

## Results (holdout test set, N = 51, threshold 0.5)

| Model | Acc | Spam F1 | Spam Recall | Ham FPR |
|---|---|---|---|---|
| **Char-CNN (proposed)** | 0.9608 | 0.9756 | 0.9756 | 0.10 |
| Naive Bayes | 0.9804 | 0.9880 | 1.0000 | 0.10 |
| Logistic Regression | 0.9608 | 0.9762 | 1.0000 | 0.20 |
| Word-CNN | 0.9412 | 0.9647 | 1.0000 | 0.30 |
| Word-LSTM | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| Char n-gram SVM | 0.9020 | 0.9425 | 1.0000 | 0.50 |

All 10 models are in `results/benchmark_holdout_results.csv`. The test set is small, so differences between most models are within the bootstrap confidence intervals reported in notebook 02; please read the absolute numbers with that in mind.

**Robustness (synthetic perturbations of test spam, plus numeric ham traps):**

| Model | Seen obf. recall | Unseen obf. recall | Numeric ham FPR | Adversarial F1 |
|---|---|---|---|---|
| Char-CNN (proposed) | 0.9919 | 1.0000 | 0.125 | 0.9879 |
| Word-CNN | 1.0000 | 1.0000 | 0.200 | 0.9840 |
| Word-LSTM | 1.0000 | 1.0000 | 0.075 | 0.9939 |

**Multi-seed stability (seeds 42-46):** accuracy 0.9490 +/- 0.0107, spam F1 0.9680 +/- 0.0069.

**Ablation (Char-CNN):** the full strategy (spam obfuscation + ham numeric injection) lowers the false-positive rate on numeric-suffix ham from 0.675 (oversampling only) to 0.125 while keeping obfuscated-spam recall at 0.9919.

## Reproducing the results

The notebooks were developed and run in Google Colab.

1. Open `01_Proposed_Model_Training.ipynb` in Google Colab.
2. Upload your corpus file `combined_dataset.xlsx` (see *Data availability*) to the Colab session.
3. Run all cells. This produces the split manifest, trained model, and tokenizer.
4. Open `02_Model_Comparison_Evaluation.ipynb` in the same session and run all cells.

All random seeds are fixed (`SEED = 42`, `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS=1`). Reported results were produced with Python 3.13, TensorFlow 2.21, scikit-learn 1.6.1, NumPy 2.4, and pandas 2.2.3. Minor numeric differences can occur across hardware and TensorFlow builds. Exact versions are listed in `requirements.txt`.

## Data availability

The SMS messages were collected from volunteers and an archive of categorized spam and contain personal information, so the raw corpus is **not redistributed**. `results/split_manifest.csv` provides the partition of every message by anonymized ID and template group, and `results/test_predictions_manifest.csv` provides per-message test predictions, so the reported metrics can be re-computed without the text.

To rerun the notebooks on your own data, provide an Excel file `combined_dataset.xlsx` (sheet `Sheet1`) with columns:

| Column | Meaning |
|---|---|
| `konten` | SMS text |
| `spam` | `SPAM` or `HAM` |
| `makna` | Spam category (`IKLAN`, `PENIPUAN`, `PROMOSI`), empty for ham/uncategorized |

## Loading the trained model

`models/tokenizer_char.pkl` stores an instance of `DeterministicCharTokenizer`, so the class (defined in notebook 01, section 5) must exist before unpickling.

```python
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# define clean_text, CHAR_TO_IDX and DeterministicCharTokenizer as in notebook 01
model = load_model("models/char_cnn_spam_model.keras")
with open("models/tokenizer_char.pkl", "rb") as f:
    tokenizer = pickle.load(f)

texts = ["selamat nomor anda memenangkan 10 juta"]
x = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=300, padding="post", truncating="post")
print(model.predict(x))  # probability of SPAM; threshold 0.5
```

## Limitations

- Small corpus (51 test messages, 10 ham), so confidence intervals are wide.
- Obfuscation in the robustness suite is rule-based and synthetic; it covers three seen and three unseen perturbation types, not all real-world evasion.
- Indonesian SMS only; the 41-character alphabet drops emoji and other symbols (2.65% of characters).

## License

Code: MIT. Model weights and manifests: for research use. (Replace with your preferred license.)

## Citation

Citation information is withheld for double-blind review and will be added after acceptance.
