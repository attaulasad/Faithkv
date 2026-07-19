# GSM8K protocol-v3 (RKV-B128) failure atlas

> This atlas analyzes a retired operating point where natural R-KV accuracy fell from 33/50 to 13/50. It is a post-hoc diagnostic analysis and cannot establish that any observed failure pattern occurs at an accuracy-preserving operating point. It generates hypotheses for later held-out testing only.

`hypothesis_status`: `not_tested` -- `operating_point_valid`: `False`

## Methodology and coordinate conventions

Each row pairs one FullKV and one R-KV base-generation record by `(source_row_index, global_seed)`. Token indices into `generated_token_ids` are 0-based. `compaction_event_steps` are recorded by `kvcot.generation.decode.generate_base` as ABSOLUTE positions in the prompt+generated stream, so `first_divergence_absolute_position = prompt_token_count + first_divergence_generated_index` before any comparison against a compaction position. See `src/kvcot/failure_atlas.py`'s module docstring for the full derivation and the mandatory prompt-offset regression test.

## Aggregate statistics

- n_pairs: 50
- retention: {'count': 50, 'mean': 0.3595886085451352, 'median': 0.34854497354497355, 'min': 0.14555555555555555, 'max': 0.6550802139037433, 'q1': 0.288972030764423, 'q3': 0.42601416853094703, 'iqr': 0.13704213776652402, 'n_below_0.50': 48, 'n_at_or_below_0.70': 50}
- compaction_count: {'count': 50, 'mean': 3.9, 'median': 4.0, 'min': 2, 'max': 7, 'q1': 3.0, 'q3': 5.0, 'iqr': 2.0}
- length_delta (rkv - full): {'count': 50, 'mean': -0.72, 'median': -13.5, 'min': -293, 'max': 374, 'q1': -75.25, 'q3': 45.25, 'iqr': 120.5}
- cap_hit: {'n_full': 0, 'n_rkv': 0}
- malformed_answer: {'n_full': 0, 'n_rkv': 0}
- malformed_think_boundary: {'n_full': 0, 'n_rkv': 0}

## Correctness-pair breakdown

- both_correct: 12
- full_only_correct: 21
- rkv_only_correct: 1
- both_wrong: 16
- FullKV accuracy: 33/50
- R-KV accuracy: 13/50

## Divergence-region breakdown (reasoning vs. answer)

{'post_think_answer': 9, 'reasoning': 41}

## Divergence relative to first compaction

{'after_first_compaction': 50}

## Identical reasoning with final-answer flips

9 of 50 pairs are token-identical through the close of the `</think>` marker (`identical_through_think`). 3 of those are correct-to-wrong flips.

| source_row_index | correctness_pair | full_len | rkv_len | first_compaction_abs | compaction_count | final_retention | think_end_idx | first_divergence_gen_idx |
|---|---|---|---|---|---|---|---|---|
| 30 | correct_wrong | 518 | 651 | 129 | 5 | 0.3123 | 235 | 237 |
| 176 | correct_correct | 312 | 339 | 129 | 3 | 0.4159 | 151 | 157 |
| 262 | correct_correct | 366 | 357 | 129 | 3 | 0.4289 | 126 | 177 |
| 271 | correct_wrong | 440 | 417 | 129 | 4 | 0.2596 | 162 | 165 |
| 491 | correct_correct | 276 | 276 | 129 | 2 | 0.6551 | 122 | 211 |
| 543 | correct_correct | 411 | 352 | 129 | 3 | 0.4302 | 145 | 158 |
| 616 | correct_correct | 325 | 312 | 129 | 3 | 0.3866 | 81 | 153 |
| 1115 | correct_wrong | 302 | 309 | 129 | 3 | 0.4037 | 107 | 148 |
| 1143 | correct_correct | 347 | 364 | 129 | 3 | 0.4302 | 126 | 189 |

### Detailed inspection: source rows 30, 271, 1115

### source row 30

- prompt_token_count: 96
- full generated length: 518; R-KV generated length: 651
- first_compaction_absolute_position: 129; compaction_count: 5; final_retention_ratio: 0.3123
- close marker (`</think>`) index -- full: 235, R-KV: 235
- first_divergence_generated_index: 237; first_divergence_absolute_position: 333
- divergence_relation_to_first_compaction: after_first_compaction; reasoning_region_category: post_think_answer
- extracted answer -- full: '109' (correct=True), R-KV: '71.\\overline{1}' (correct=False)
- decoded text after `</think>` -- full: "\n\nLet's solve the problem step by step.\n\n**Step 1: Define the Variables**\n\nLet Darrell's current age be \\(7x\\) and Allen's current age be \\(11x\\). The ratio of their ages is given as 7:11.\n\n**Step 2: Set Up the Equation Based on Total Age**\n\nThe sum of their ages is 162 years:\n\\[\n7x + 11x = 162\n\\]\n\n**Step 3: Solve for \\(x\\)**\n\nCombine like terms:\n\\[\n18x = 162\n\\]\n\nDivide both sides by 18:\n\\[\nx = \\f"
- decoded text after `</think>` -- R-KV: "\n\nTo determine Allen's age 10 years from now, follow these steps:\n\n1. **Understand the Given Information:**\n   - Darrell and Allen's current ages are in the ratio of \\( 7:11 \\).\n   - The total of their current ages is \\( 162 \\) years.\n\n2. **Set Up Variables:**\n   - Let \\( D \\) represent Darrell's current age.\n   - Let \\( A \\) represent Allen's current age.\n   - According to the ratio, \\( D:A = 7:1"

### source row 271

- prompt_token_count: 104
- full generated length: 440; R-KV generated length: 417
- first_compaction_absolute_position: 129; compaction_count: 4; final_retention_ratio: 0.2596
- close marker (`</think>`) index -- full: 162, R-KV: 162
- first_divergence_generated_index: 165; first_divergence_absolute_position: 269
- divergence_relation_to_first_compaction: after_first_compaction; reasoning_region_category: post_think_answer
- extracted answer -- full: '20' (correct=True), R-KV: '12' (correct=False)
- decoded text after `</think>` -- full: '\n\n**Step-by-Step Solution:**\n\n1. **Determine the number of bedrooms in both houses:**\n   - Each house has **3 bedrooms**.\n   - There are **2 houses**.\n   - **Total bedrooms** = \\(2 \\text{ houses} \\times 3 \\text{ bedrooms/house} = 6 \\text{ bedrooms}\\).\n\n2. **Calculate the number of windows in the bedrooms:**\n   - Each bedroom has **2 windows**.\n   - **Total windows in bedrooms** = \\(6 \\text{ bedroo'
- decoded text after `</think>` -- R-KV: "\n\n**Solution:**\n\nLet's break down the problem step by step.\n\n1. **Number of Bedrooms:**\n   - Each house has **2 bedrooms**.\n   - There are **2 houses**.\n   - **Total number of bedrooms** = \\(2 \\text{ bedrooms/house} \\times 2 \\text{ houses} = 4 \\text{ bedrooms}\\).\n\n2. **Windows in Bedrooms:**\n   - Each bedroom has **2 windows**.\n   - **Total number of windows in bedrooms** = \\(4 \\text{ bedrooms} \\t"

### source row 1115

- prompt_token_count: 123
- full generated length: 302; R-KV generated length: 309
- first_compaction_absolute_position: 129; compaction_count: 3; final_retention_ratio: 0.4037
- close marker (`</think>`) index -- full: 107, R-KV: 107
- first_divergence_generated_index: 148; first_divergence_absolute_position: 271
- divergence_relation_to_first_compaction: after_first_compaction; reasoning_region_category: post_think_answer
- extracted answer -- full: '5' (correct=True), R-KV: '6' (correct=False)
- decoded text after `</think>` -- full: "\n\n**Solution:**\n\n1. **Determine the initial amount spent:**\n   \n   Anna had a budget of \\$16 and had already spent \\$4.\n   \\[\n   \\text{Remaining after initial spending} = \\$16 - \\$4 = \\$12\n   \\]\n\n2. **Calculate the amount spent today:**\n   \n   After today's purchases, Anna has \\$2 left in her budget.\n   \\[\n   \\text{Amount spent today} = \\$12 - \\$2 = \\$10\n   \\]\n\n3. **Find the cost of each book:**\n "
- decoded text after `</think>` -- R-KV: '\n\n**Solution:**\n\n1. **Determine the initial amount spent:**\n   \n   Anna had a budget of \\$16 and had already spent \\$4.\n   \\[\n   \\text{Amount spent so far} = \\$16 - \\$4 = \\$12\n   \\]\n\n2. **Calculate the remaining budget:**\n   \n   After spending \\$12, Anna has \\$2 left.\n   \\[\n   \\text{Remaining budget} = \\$12 - \\$10 = \\$2\n   \\]\n   (Note: Since she bought two books, the total spent is \\$12, leaving h'

## Limitations and claim boundary

- This is a 50-example post-hoc diagnostic over a RETIRED operating point (accuracy dropped 0.40, exceeding the frozen 0.10 ceiling). No causal claims are supported by any correlation reported here.
- Per-record retention is a single END-OF-GENERATION snapshot (`RetentionSummary.instantaneous_retention_ratio`); no per-step retention trajectory is committed, so `final_retention_ratio` is the only retention statistic directly available per pair -- not a minimum or a trajectory.
- "Decoded" context fields are derived from each record's own already-decoded `decoded_output` string; no tokenizer was loaded or downloaded to produce them. The raw-token-id window fields are token ids, not decoded text -- decoding a token window precisely would require the pinned tokenizer, which this CPU-only, no-network analysis deliberately does not load.

This measures counterfactual behavioral dependence on the visible, generated chain-of-thought under truncation — NOT internal faithfulness, NOT whether reasoning is 'real', and NOT a claim about the model's internal cognition. See docs/EXPERIMENT.md and build brief §1.
