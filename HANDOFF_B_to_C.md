# Handoff: Person B to Person C
Person B finished the models. This note says what you get, how to run it, and what you must say in the app and report.

## 1. What you get
- Fine-tuned CodeBERT: the `best` folder (about 500 MB, NOT in GitHub). Download: https://drive.google.com/drive/folders/1npGJ1ChxLWMGug-BBoOBbiDSyPbFxHUu?usp=sharing. It holds the weights, the tokenizer and `codebert_config.json` (threshold, max_length, preprocessing).
- Baseline (generic TF-IDF + LogReg): `models/baseline_generic/` (in GitHub).
- Results: `reports/before_after.md` and `reports/before_after.csv` (all tables, slices, CVEfixes, pair check).
- Script that works without Flask: `Scripts/predict_codebert.py`.

## 2. First check (gate)
    pip install -r requirements.txt
    pip install torch transformers
    python Scripts/predict_codebert.py --model-dir <path to best> --code "void f(char *s){ char b[8]; strcpy(b, s); }"
It must print one line: model score, threshold 0.66, and "potentially vulnerable" or "not flagged".

## 3. Rules for inference (the app must follow these)
- Apply `normalize_code` (in `predict_codebert.py`) to the pasted code BEFORE the model. Training used it. Do not use any other cleaning.
- max_length is 512 tokens (510 code tokens). If a function is longer, only the start is analysed: show a warning (the script prints one).
- Frozen threshold = 0.66 (in `codebert_config.json`). Score >= 0.66 means "potentially vulnerable", otherwise "not flagged".
- Call the output a **model score**, never a probability or "x% certain". "Not flagged" does not mean secure.
- Load the model once at app start. Never execute the submitted code. Escape it when showing it.

## 4. Key numbers (BigVul test, opened once; 27,720 functions, 5.2% vulnerable)
See `reports/before_after.md` for the full tables. Short version:
- Majority guess: accuracy 0.948 but it finds nothing (recall 0).
- CodeBERT before training: PR-AUC 0.045 (random). After: test PR-AUC 0.151, F1 0.223, recall 0.327, precision 0.169.
- Generic TF-IDF + LogReg: test PR-AUC 0.159, F1 0.212. The difference to CodeBERT is not significant (bootstrap interval includes 0).
- CVEfixes (outside data): both models are at chance (ROC-AUC about 0.50). On before-vs-after-fix pairs they rank the fixed version higher more often than the vulnerable one.
- So the honest message is: fine-tuning helped (random to above chance), but the detector is weak and does not transfer. Do not oversell it in the demo.

## 5. Known problems (say them in the report)
- BigVul whitespace shortcut: 56.1% of vulnerable vs 1.9% of safe training functions have trailing whitespace. A first CodeBERT run on raw code showed val PR-AUC 0.864 because of this. The final model is retrained on normalised code, which is why normalisation is mandatory at inference.
- Training used a subsample (46,748 of 129,360 rows), one run, 2 epochs.
- 13.0% of normalised test functions (18.6% of raw training functions) exceed 512 tokens; vulnerable ones are cut more often.
- Length is a partial shortcut (long functions are more often vulnerable).
- Labels are noisy (BigVul from commits; CVEfixes "before a fix" is not proven vulnerable).
- Open with the instructor: is the BigVul release acceptable as "raw", does 100,000 rows mean total or training rows, is training on a GPU subsample acceptable.

## 6. Rules still in force
- Do not retune on the test set. Do not train on CVEfixes. No synthetic data.
- Keep data and model weights out of GitHub.
- If you retrain or change preprocessing, re-run the evaluation and tell Person B.