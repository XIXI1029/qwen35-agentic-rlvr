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
