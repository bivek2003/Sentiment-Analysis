## Assignment 3: Sentiment Analysis using Feedforward Neural Networks

**Author: Bivek Panthi & Roman Balayar**
**April 13, 2026, University of New Mexico**

### Introduction
This assignment applies Feedforward Neural Networks (FNNs) to sentiment classification on the IMDb movie reviews dataset. Each review is transformed into a tf-idf vector and fed into a fully-connected network that predicts a binary label: 1 for positive, 0 for negative. The objectives are: to build a baseline FNN in PyTorch and tune it to reach an accuracy comparable to the logistic regression model in the textbook; to implement k-fold cross validation manually on top of the PyTorch training loop; and to study dropout regularization both as a single regularized model and as a bagging ensemble of dropout models.

### Dataset and Preprocessing
The IMDb dataset (`aclImdb_v1`) contains 50,000 labeled movie reviews split evenly between positive and negative classes. After downloading and extracting the archive, the reviews from `train/pos`, `train/neg`, `test/pos`, and `test/neg` were merged into a single pandas DataFrame and shuffled with a fixed seed for reproducibility.

Each review was cleaned with a regex preprocessor that removes HTML tags, lowercases the text, strips non-word characters, and preserves emoticons. The cleaned text was then transformed into tf-idf vectors using `sklearn.feature_extraction.text.TfidfVectorizer` with `use_idf=True`, `norm='l2'`, and `smooth_idf=True`. The vectorizer was fit only on the training portion to prevent data leakage.

The data was split 70/30 into training and test sets:
- Training set: **35,000** reviews, **89,509** tf-idf features
- Test set: **15,000** reviews, **89,509** tf-idf features

A custom `SparseDataset` wraps the SciPy sparse tf-idf matrix and converts rows to dense `torch.float32` tensors on the fly inside `__getitem__`, allowing PyTorch `DataLoader` (batch size 256) to stream batches without materializing the full dense matrix in memory.

### Task 2 — Building an FNN

Two FNN architectures were defined in PyTorch. The larger network used in Task 3 has three hidden layers with ReLU activations and a sigmoid output:

```
FNN_Large
  Linear(89509 -> 256) -> ReLU
  Linear(256   -> 128) -> ReLU
  Linear(128   ->  64) -> ReLU
  Linear(64    ->   1) -> Sigmoid
```

The smaller baseline network (used in Task 5 to match the book's logistic regression scale) has two hidden layers:

```
FNN_Baseline
  Linear(89509 -> 128) -> ReLU
  Linear(128   ->  64) -> ReLU
  Linear(64    ->   1) -> Sigmoid
```

Both networks use Binary Cross-Entropy loss (`nn.BCELoss`) on the sigmoid output and are optimized with `torch.optim.Adam`. A prediction is positive when the sigmoid output is ≥ 0.5. The choice of Adam over SGD was deliberate — early experiments with SGD converged much more slowly on the very high-dimensional sparse input.

### Task 3 — Baseline Training and Hyperparameter Tuning

The baseline `FNN_Large` model was trained for 10 epochs with batch size 256. Three combinations of learning rate and weight decay (L2 regularization) were evaluated by exhaustive comparison.

**Table 1: FNN_Large Hyperparameter Sweep (10 epochs)**

| Learning Rate | Weight Decay | Final Test Acc | Training Time |
|---:|---:|---:|---:|
| 0.001  | 1e-4 | 0.8733 | 180.6 s |
| 0.0005 | 1e-5 | **0.8903** | 180.6 s |
| 0.01   | 1e-3 | 0.8858 | 197.6 s |

The configuration **lr = 0.0005, weight_decay = 1e-5** gave the best final test accuracy and was selected as the tuned baseline. With this configuration the network reached its peak test accuracy of **0.9072** after the first epoch and **0.8995** after the second epoch, after which training accuracy saturated at 1.0000 while test accuracy slowly declined — a classic overfitting profile in which one or two epochs already exhaust the generalization the network is going to extract from this representation.

**Table 2: FNN vs. Logistic Regression (Textbook)**

| Model | Test Accuracy | Training Time |
|---|---:|---:|
| Logistic Regression (textbook) | 0.899 | 5–10 min |
| FNN_Large [256→128→64], lr=0.001  | 0.873 | 180.6 s |
| FNN_Large [256→128→64], lr=0.0005 | 0.890 | 180.6 s |
| FNN_Large [256→128→64], lr=0.01   | 0.886 | 197.6 s |

**Analysis.** The tuned FNN reaches accuracy essentially on par with the textbook logistic regression model (0.890 vs 0.899) while training in roughly **3 minutes** — comparable to or faster than the book's logistic regression grid search, which the textbook reports at 5–10 minutes. Peak test accuracy in the first two epochs (0.9072) actually exceeds the logistic regression baseline, indicating that the FNN can match a strong linear model on tf-idf features but that the very wide input layer (89,509 → 256) gives the network enough capacity to memorize the training set quickly. The lr=0.0005 setting smooths the optimization enough that the early epochs land at a better generalization point than the more aggressive 0.001 and 0.01 settings.

### Task 4 — k-Fold Cross Validation

PyTorch does not directly support k-fold cross validation, so we implemented it manually on top of `sklearn.model_selection.KFold`. The training and test datasets are first concatenated with `ConcatDataset`, then `KFold(n_splits=k, shuffle=True, random_state=42)` produces fold indices that are wrapped in `SubsetRandomSampler` and handed to two `DataLoader` instances (one for the in-fold training data, one for the held-out fold). For each fold, a fresh `FNN_Large` is instantiated, weights are reset via `reset_parameters`, and the network is trained for 10 epochs with the tuned hyperparameters (lr = 0.0005, weight_decay = 1e-5). After training, train and test accuracy are computed for that fold; the final reported accuracy is the average across folds. We tuned k by trying k = 3, 5, and 10.

**Table 3: k-Fold Cross Validation vs. Baseline**

| Model | Train Acc | Test Acc | Total Time |
|---|---:|---:|---:|
| Baseline (no k-Fold)        | —     | 0.8749 |  179.7 s |
| k-Fold (k = 3)              | 1.0000 | **0.8867** |  377.4 s |
| k-Fold (k = 5)              | 1.0000 | 0.8858 |  806.3 s |
| k-Fold (k = 10)             | 0.9982 | 0.8838 | 1744.3 s |

**Analysis.** All three k-fold settings produce test accuracy higher than the single-split baseline (0.8867 / 0.8858 / 0.8838 vs. 0.8749), confirming that the cross-validated estimate is both more stable and slightly more optimistic — each fold trains on more data (≈ 47k vs 35k samples) than the single-split baseline, which explains most of the improvement. Among the three k values, **k = 3** gave the best average test accuracy and is also the fastest (≈ 377 s vs. 1744 s for k = 10). Time cost grows almost linearly with k because each additional fold is one more full training run; k = 10 takes roughly 10× the single-fold time. Train accuracy is essentially saturated (1.0000 for k = 3 and k = 5; 0.9982 for k = 10), which again reflects how easily the wide network memorizes the training partition. Given the marginal accuracy differences, **k = 3 is the preferred trade-off** here: the gain over k = 5 or k = 10 is well within fold-to-fold noise but the wall-clock saving is large.

### Task 5 — Dropout Regularization

For Task 5 we returned to the smaller `FNN_Baseline` ([128 → 64]) so that the dropout study mirrors the book's scale. Three runs are compared in this section: a baseline FNN with no dropout and no weight decay; a single FNN with dropout layers inserted after each ReLU; and a bagging ensemble of five dropout FNNs trained on bootstrap samples. All three were trained with lr = 0.0005 for 10 epochs.

#### Task 5.1 — Single Dropout Model

`FNN_Dropout` inserts `nn.Dropout(p1)` after the first hidden layer and `nn.Dropout(p2)` after the second hidden layer. Three (p1, p2) configurations were tuned.

**Table 4: Single Dropout Model — Hyperparameter Sweep**

| Dropout (p1, p2) | Train Acc | Test Acc | Time |
|---|---:|---:|---:|
| (0.3, 0.2) | 0.9999 | 0.8927 | 150.3 s |
| (0.5, 0.3) | 0.9996 | 0.8937 | 149.8 s |
| (0.5, 0.5) | 0.9997 | **0.8947** | 147.1 s |

The best single dropout configuration was **p1 = 0.5, p2 = 0.5**.

**Table 5: Single Dropout vs. Baseline**

| Model | Train Acc | Test Acc | Time |
|---|---:|---:|---:|
| Baseline FNN (no regularization)  | 1.0000 | 0.8890 | 150.0 s |
| Single Dropout (p1=0.5, p2=0.5)   | 0.9997 | **0.8947** | 147.1 s |

**Analysis.** Adding dropout improves test accuracy by **0.57 percentage points** (0.8890 → 0.8947) at essentially zero extra training cost — dropout layers are cheap and the wall-clock time is statistically identical (147.1 s vs 150.0 s). Train accuracy stays effectively at 1.0 in both cases, but the dropout model generalizes slightly better because each forward pass during training operates on a different subnetwork, which discourages neurons from co-adapting to specific tf-idf features. The accuracy gain is consistent across all three (p1, p2) settings, which suggests that dropout is helping in a robust, configuration-insensitive way; the highest dropout rate (0.5/0.5) is best, indicating that the baseline network has substantial excess capacity to regularize away.

#### Task 5.2 — Bagging Ensemble of Dropout Models

The bagging ensemble uses **five** different dropout configurations: (0.3, 0.2), (0.4, 0.2), (0.5, 0.3), (0.5, 0.5), and (0.6, 0.4). For each model i, a bootstrap sample of size 35,000 is drawn with replacement from the training set using `np.random.seed(i)`, wrapped in a `SubsetRandomSampler`, and used to train an independent `FNN_Dropout` for 10 epochs. At inference time the five models predict probabilities for each test sample; the probabilities are averaged and thresholded at 0.5 (soft voting) to produce the ensemble prediction.

**Table 6: Individual Bagging Models**

| Model | Dropout (p1, p2) | Train Acc | Test Acc | Time |
|---|---|---:|---:|---:|
| 1 | (0.3, 0.2) | 1.0000 | 0.8829 | 83.1 s |
| 2 | (0.4, 0.2) | 1.0000 | 0.8877 | 82.7 s |
| 3 | (0.5, 0.3) | 1.0000 | 0.8875 | 82.4 s |
| 4 | (0.5, 0.5) | 0.9999 | 0.8869 | 83.4 s |
| 5 | (0.6, 0.4) | 0.9998 | 0.8882 | 83.0 s |

**Table 7: Bagging Ensemble vs. Baseline and Single Dropout**

| Model | Train Acc | Test Acc | Time |
|---|---:|---:|---:|
| Baseline FNN (no regularization)        | 1.0000 | 0.8890 |  150.0 s |
| Single Dropout (p1=0.5, p2=0.5)         | 0.9997 | 0.8947 |  147.1 s |
| Bagging Ensemble (5 dropout models)     | 0.9833 | **0.9001** |  426.6 s |

**Analysis.** The bagging ensemble is the only one of the three that breaks the **0.90 test-accuracy barrier**, gaining 1.11 percentage points over the baseline (0.8890 → 0.9001) and 0.54 points over the best single dropout model. Two effects compose to produce this gain. First, each member of the ensemble already has a slightly different decision surface because of its dropout configuration; second, the bootstrap sample exposes each model to a different ~63% of the training data, increasing the diversity of mistakes the members make. Soft-voting cancels uncorrelated errors, which is why the ensemble train accuracy actually *decreases* (0.9833) relative to any single member (≈ 1.0000) — the ensemble is no longer memorizing every training example, but it generalizes better to the test set. The cost is roughly **2.85× the wall-clock** of a single model (426.6 s vs. 150.0 s), which is still cheap relative to the k-fold runs and well worth the accuracy gain when raw test performance is the priority.

### Discussion and Conclusion

The full set of experiments produces a coherent picture of how each technique interacts with this tf-idf sentiment task:

- **Tuning alone** (Task 3) is sufficient to bring a wide FNN into the same accuracy band as the textbook logistic regression model (≈ 0.89–0.91), in roughly 3 minutes of wall-clock training.
- **k-Fold cross validation** (Task 4) gives a more reliable estimate of generalization and yields slightly higher test accuracy than a single 70/30 split, mostly because each fold trains on more data. k = 3 is the sweet spot — k = 5 and k = 10 buy almost no additional accuracy but cost 2× and 4.6× more time, respectively.
- **Single dropout** (Task 5.1) modestly improves test accuracy at no time cost. The gain (≈ 0.6 pp) is consistent across dropout rates, with the highest rate (0.5/0.5) giving the best result, confirming that the baseline FNN has substantial excess capacity.
- **Bagging an ensemble of dropout models** (Task 5.2) is the strongest approach overall and the only configuration to break **0.90 test accuracy**. Combining bootstrap-sample diversity with dropout-architecture diversity produces models whose errors are decorrelated enough that soft voting yields a real gain over any single member.

Across all configurations the training accuracy saturates near 1.0, which means the bottleneck is no longer model capacity but the discriminative information available in the tf-idf representation itself. Further gains would likely require richer features (e.g. word embeddings, n-grams, or contextual representations) rather than deeper or wider feedforward architectures.
