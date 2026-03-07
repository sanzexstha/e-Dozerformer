# mask_comp_v1 — Mask Type Comparison

**Date:** 2026-02-19  
**Goal:** Evaluate different attention sparsity patterns in DozerAttention.

## Masks Compared
- `extreme_mask`: Pure label-based attention. Query attends only to tokens with same 0/1 label
- `dozer`: Standard
- `dozer_ext_only`: Normal query attend to Dozer keys, extreme query attends to dozer keys and extreme keys
- `dozer_ext_0`: Normal query attends to keys from Dozer and Extreme after AND operator, key has to be True in both attention matrix to be select.
- `dozer_ext_null`: Dozer mask restricting attention to region of extreme mask allowed to extreme token
- `dozer_AND_ext`: When both dozer and extreme mask agrees (1)

## Findings
In general, dozer mask still performs better than rest of the sparsity pattern.

## Next
→ 0/1 labels could alter the performance, so we can experiment with GDN to get different version of labels.