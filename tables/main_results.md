| Layer (task, n) | Base | Our RL | Strong ref. | Verifiable reward |
|---|---|---|---|---|
| Reasoning (GSM8K, 50) | 30.0% | 50.0% | 40.0% | verifier: numeric match |
| Execution (HumanEval, 164) | 14.6% | 25.0% | -- | verifier: sandboxed unit tests |
| Selection (BFCL v3, 400) | 0.207 | 0.795 | 0.590 | verifier: skill-set match + refusal |
