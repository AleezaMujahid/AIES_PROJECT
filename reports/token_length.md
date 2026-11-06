# CodeBERT token-length check

Tokenizer: `microsoft/codebert-base`. Limit: 512 tokens (including the 2 special tokens). Rows measured: 30,000 (random sample of up to 30,000 per split).

| Group | Functions | Median | 90th pct | 95th pct | 99th pct | Max | Longer than 512 |
|---|---|---|---|---|---|---|---|
| all | 30,000 | 170 | 854 | 1423 | 3899 | 119,343 | 18.6% |
| split: train | 30,000 | 170 | 854 | 1423 | 3899 | 119,343 | 18.6% |
| label 0 (safe) | 28,463 | 162 | 782 | 1262 | 3130 | 119,343 | 17.1% |
| label 1 (vulnerable) | 1,537 | 453 | 2852 | 5119 | 15154 | 90,254 | 46.2% |
| without chrome and linux | 10,197 | 238 | 1371 | 2323 | 7413 | 119,343 | 28.6% |

## Does the cut hit vulnerable functions more?

- Vulnerable share among all measured functions: **5.1%**
- Vulnerable share among functions that get cut: **12.7%**
- Vulnerable share among functions that fit: **3.4%**

## What this means

- Functions over 512 tokens are truncated: CodeBERT sees only the first part and never the rest.
- If a vulnerable line sits after the cut, the model cannot see it. Say this in the report as a limitation.
- Use `max_length = 512` for fine-tuning (the model's hard limit). Longer is not possible without chunking.