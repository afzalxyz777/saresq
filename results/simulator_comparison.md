# Simulator: policy comparison (100 trials/policy)

| policy | survivors_confirmed | distractors_confirmed | survivors_missed | area_covered_m2 | battery_remaining_pct | pct_trials_first_confirmed | time_to_first_confirmation_s (when it happens) | pct_trials_all_confirmed | time_to_confirm_all_s (when it happens) |
|---|---|---|---|---|---|---|---|---|---|
| lawnmower | 0.42 ± 0.13 | 0.12 ± 0.07 | 1.56 ± 0.22 | 30000.00 ± 0.00 | 47.29 ± 0.92 | 34% | 154.48 ± 28.31 | 0% | never |
| adaptive | 1.71 ± 0.21 | 1.54 ± 0.21 | 1.87 ± 0.19 | 26640.60 ± 555.29 | 1.12 ± 0.50 | 92% | 109.15 ± 15.15 | 2% | 270.96 ± 37.58 |
| lawnmower_low | 0.55 ± 0.16 | 0.17 ± 0.07 | 1.17 ± 0.20 | 30000.00 ± 0.00 | 16.61 ± 1.04 | 38% | 197.86 ± 41.11 | 0% | never |

Wall time for 100 trials x 3 policies: 0.09s
