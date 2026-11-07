# Before vs after fine-tuning: results (Person B)

All numbers come from saved files in `codesentinel_ckpt_v2` (validation scores, test predictions, CVEfixes predictions).
Output is a **model score**, not a probability. Test set was opened once; the epoch and the threshold were chosen on validation only.

## 1. Setup
- CodeBERT (`microsoft/codebert-base`) + new 2-class head, max_length 512 (keep start), lr 2e-5, batch 16, 2 epochs, seed 42.
- Training subsample: 46,748 of 129,360 BigVul training rows (all 6,748 vulnerable + 40,000 safe).
  Class weights [0.584, 3.464] (balanced for the subsample).
- Input is normalised code: per-line strip, spaces/tabs collapsed, blank lines dropped (`normalize_code`). Frozen CodeBERT threshold 0.66 (best F1 on validation). Best epoch: 2.
- Baseline: generic-token TF-IDF + Logistic Regression, trained on all 129,360 rows, its own saved threshold 0.637.

## 2. Validation (CodeBERT, before vs after training)
| stage | threshold_type | accuracy | precision | recall | f1 | pr_auc | roc_auc |
|---|---|---|---|---|---|---|---|
| CodeBERT before (epoch 0) | fixed 0.5 | 0.948 | 0.000 | 0.000 | 0.000 | 0.046 | 0.476 |
| CodeBERT before (epoch 0) | best F1 on val | 0.127 | 0.053 | 0.938 | 0.101 | 0.046 | 0.476 |
| CodeBERT after epoch 1 | fixed 0.5 | 0.790 | 0.137 | 0.571 | 0.221 | 0.148 | 0.759 |
| CodeBERT after epoch 1 | best F1 on val | 0.871 | 0.172 | 0.387 | 0.239 | 0.148 | 0.759 |
| CodeBERT after epoch 2 | fixed 0.5 | 0.829 | 0.154 | 0.508 | 0.237 | 0.156 | 0.768 |
| CodeBERT after epoch 2 | best F1 on val | 0.877 | 0.181 | 0.385 | 0.246 | 0.156 | 0.768 |

## 3. Test set (BigVul, 27,720 rows, opened once)
| model | stage | threshold | accuracy | precision | recall | f1 | pr_auc | roc_auc |
|---|---|---|---|---|---|---|---|---|
| Majority guess | before | 0.500 | 0.948 | 0.000 | 0.000 | 0.000 | 0.052 | 0.500 |
| TF-IDF + LogReg (generic) | after (its saved threshold) | 0.637 | 0.882 | 0.163 | 0.305 | 0.212 | 0.159 | 0.720 |
| CodeBERT | before (start weights, thr 0.5) | 0.500 | 0.948 | 0.000 | 0.000 | 0.000 | 0.045 | 0.466 |
| CodeBERT | after (best epoch, frozen thr) | 0.660 | 0.881 | 0.169 | 0.327 | 0.223 | 0.151 | 0.740 |

Confusion matrices:

| model | stage | TN | FP | FN | TP |
|---|---|---|---|---|---|
| Majority guess | before | 26274 | 0 | 1446 | 0 |
| TF-IDF + LogReg (generic) | after (its saved threshold) | 24006 | 2268 | 1005 | 441 |
| CodeBERT | before (start weights, thr 0.5) | 26274 | 0 | 1446 | 0 |
| CodeBERT | after (best epoch, frozen thr) | 23954 | 2320 | 973 | 473 |

Accuracy of the majority guess is high only because about 94.8% of functions are safe; it finds no vulnerabilities.

## 4. Slices of the test set
| slice | rows | % vulnerable | TF-IDF PR-AUC | TF-IDF recall | TF-IDF F1 | CodeBERT PR-AUC | CodeBERT recall | CodeBERT F1 |
|---|---|---|---|---|---|---|---|---|
| all test rows | 27720 | 5.200 | 0.159 | 0.305 | 0.212 | 0.151 | 0.327 | 0.223 |
| without chrome+linux | 8410 | 6.200 | 0.238 | 0.508 | 0.275 | 0.227 | 0.498 | 0.287 |
| only chrome+linux | 19310 | 4.800 | 0.108 | 0.190 | 0.158 | 0.114 | 0.230 | 0.175 |
| fits in 512 tokens | 24103 | 3.800 | 0.077 | 0.118 | 0.111 | 0.082 | 0.154 | 0.128 |
| cut at 512 tokens | 3617 | 14.400 | 0.271 | 0.637 | 0.302 | 0.287 | 0.635 | 0.326 |

CodeBERT minus TF-IDF, PR-AUC: mean -0.007, 95% interval [-0.019, +0.006]
CodeBERT minus TF-IDF, F1: mean +0.011, 95% interval [-0.004, +0.027]
(bootstrap over rows, 500 resamples; resampling commits instead would give wider intervals)

## 5. Outside test: CVEfixes (never used for training or tuning)
Vulnerable share is 39.9% (full) and 49.9% (strict), so chance PR-AUC is 0.40 and 0.50.

| set | model | accuracy | precision | recall | f1 | pr_auc | roc_auc |
|---|---|---|---|---|---|---|---|
| CVEfixes full | Majority guess | 0.601 | 0.000 | 0.000 | 0.000 | 0.399 | 0.500 |
| CVEfixes full | TF-IDF + LogReg (generic) | 0.523 | 0.398 | 0.379 | 0.389 | 0.401 | 0.506 |
| CVEfixes full | CodeBERT (after) | 0.526 | 0.396 | 0.358 | 0.376 | 0.392 | 0.501 |
| CVEfixes strict | Majority guess | 0.501 | 0.000 | 0.000 | 0.000 | 0.499 | 0.500 |
| CVEfixes strict | TF-IDF + LogReg (generic) | 0.491 | 0.488 | 0.422 | 0.453 | 0.493 | 0.490 |
| CVEfixes strict | CodeBERT (after) | 0.497 | 0.495 | 0.401 | 0.443 | 0.494 | 0.490 |

| set | model | TN | FP | FN | TP |
|---|---|---|---|---|---|
| CVEfixes full | Majority guess | 7155 | 0 | 4754 | 0 |
| CVEfixes full | TF-IDF + LogReg (generic) | 4426 | 2729 | 2950 | 1804 |
| CVEfixes full | CodeBERT (after) | 4558 | 2597 | 3051 | 1703 |
| CVEfixes strict | Majority guess | 2638 | 0 | 2628 | 0 |
| CVEfixes strict | TF-IDF + LogReg (generic) | 1473 | 1165 | 1518 | 1110 |
| CVEfixes strict | CodeBERT (after) | 1562 | 1076 | 1575 | 1053 |

## 6. Before-vs-after-fix pair check (same function, same commit)
Label 1 is the version before the fix, label 0 the version after. "score before > after" is about 0.5 for a model that cannot tell them apart.

| set | model | pairs | score before > after | correct (before flagged, after not) | both flagged | neither flagged | reversed (after flagged, before not) |
|---|---|---|---|---|---|---|---|
| CVEfixes strict | TF-IDF + LogReg (generic) | 2548 | 0.345 | 0.011 | 0.418 | 0.539 | 0.032 |
| CVEfixes strict | CodeBERT (after) | 2548 | 0.336 | 0.022 | 0.383 | 0.566 | 0.030 |
| CVEfixes full | TF-IDF + LogReg (generic) | 4145 | 0.322 | 0.009 | 0.385 | 0.571 | 0.035 |
| CVEfixes full | CodeBERT (after) | 4145 | 0.366 | 0.025 | 0.339 | 0.607 | 0.029 |

## 7. Earlier run that exposed a shortcut (kept as a finding)
BigVul has a whitespace artifact: 56.1% of vulnerable but only 1.9% of safe training functions contain a line with trailing whitespace.
A first CodeBERT run on raw code reached validation PR-AUC 0.864 (F1 0.81). A model using only formatting features reached PR-AUC 0.632 on validation,
and after whitespace normalisation formatting plus length fell to 0.175. CodeBERT was therefore retrained on normalised code (section 2). The TF-IDF baseline ignores whitespace and was not affected.

## 8. Limitations to state in the report
- Training used a subsample (46,748 of 129,360 rows); one run, one configuration, 2 epochs.
- Inputs longer than 512 tokens are cut (18.6% of raw training functions, 13.0% of normalised test functions); vulnerable functions are cut more often.
- Longer functions are more often vulnerable, so part of the score can come from length.
- BigVul labels are noisy and project-skewed (chrome and linux dominate). CVEfixes labels mean "before a fix", not proven vulnerable.
- Near-duplicate analysis was a spot check only (nearest-neighbour TF-IDF similarity; no clear effect on recall).
- Thresholds were chosen on validation; scores are not calibrated probabilities. "Not flagged" does not mean secure.
