# 科研格式表格（自动生成）

> 由 `tables/make_tables.py` 从 `evidence/results/*.json` 生成；重跑评测后重新执行即可更新。

## Table 1. Three-layer summary

| Layer (task, n) | Base | Our RL | Strong ref. | Verifiable reward |
|---|---|---|---|---|
| Reasoning (GSM8K, 50) | 30.0% | 50.0% | 40.0% | verifier: numeric match |
| Execution (HumanEval, 164) | 14.6% | 25.0% | -- | verifier: sandboxed unit tests |
| Selection (BFCL v3, 400) | 0.207 | 0.795 | 0.590 | verifier: skill-set match + refusal |

## Table 2. Reasoning layer (GSM8K)

| Model | Accuracy | Correct |
|---|---|---|
| 4B-Base (no RL) | 30.0% | 15/50 |
| 4B-SFT (cold start) | 34.0% | 17/50 |
| 4B-GRPO v1 (G=4, binary reward) | 34.0% | 17/50 |
| 4B-GRPO v2 (G=8, partial credit) | 50.0% | 25/50 |
| 9B-Base (reference) | 40.0% | 20/50 |

## Table 3. Execution layer (HumanEval)

| Model | pass@1 | Passed |
|---|---|---|
| 4B-Base | 14.6% | 24/164 |
| 4B-SFT (math cold start) | 21.9% | 36/164 |
| 4B-Code-GRPO v1 | 21.9% | 36/164 |
| 4B-Code-GRPO v2 | 25.0% | 41/164 |

## Table 4. Selection layer (BFCL v3, stratified 400)

| Strategy / model | Exact | F1 | False-load↓ | Miss↓ | Avg. loads |
|---|---|---|---|---|---|
| Load all candidates | 0.200 | 0.492 | 1.000 | 0.000 | 1.84 |
| Random pick | 0.330 | 0.412 | 1.000 | 0.000 | 1.00 |
| Similarity retrieval (common practice) | 0.485 | 0.606 | 1.000 | 0.000 | 1.00 |
| Retrieval + threshold rule | 0.590 | 0.658 | 0.650 | 0.154 | 0.77 |
| 4B-Base (no RL) | 0.207 | 0.301 | 0.894 | 0.100 | 2.17 |
| Math-RL only (no skill RL) | 0.160 | 0.311 | 0.881 | 0.142 | 1.97 |
| Skill-RL from Base | 0.360 | 0.341 | 0.656 | 0.121 | 1.76 |
| Skill-RL from math-RL checkpoint | 0.795 | 0.652 | 0.225 | 0.067 | 1.00 |

## Table 5. Selection layer, per category

| Category | Retrieval | Rule | Base | Skill-RL (ours) |
|---|---|---|---|---|
| multiple | 0.887 | 0.863 | 0.013 | 0.850 |
| live_multiple | 0.537 | 0.400 | 0.062 | 0.625 |
| simple | 1.000 | 0.988 | 0.750 | 0.950 |
| irrelevance | 0.000 | 0.325 | 0.150 | 0.650 |
| live_irrelevance | 0.000 | 0.375 | 0.062 | 0.900 |

## Table 6. Cross-layer transfer

| Setting | Before skill RL | After skill RL | Gain |
|---|---|---|---|
| Start = Base | 0.207 | 0.360 | +0.152 |
| Start = math-RL checkpoint | 0.160 | 0.795 | +0.635 |
| Transfer effect Δ(mathRL) − Δ(base) | -- | -- | +0.483 |
