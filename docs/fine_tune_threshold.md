## fine_tune_threshold

**Date:** 2026-03-4  
**Goal:** To fine-tune the patch threshold parameter that labels a patch either normal or extreme based on the number of 1's present within the patch 

### Sparse Mask Compared
- `dozer_ext_0`: Normal queries attend to only dozer keys but not extreme keys i.e $ \text{DozerMask} \cap \text{ExtremeMask} $. Extreme queries attend to only extreme keys. Let $M_D$ be the Dozer mask and $M_E$ be the Extreme mask.

$$
M_{ij} =
\begin{cases}
M_D(i,j) \land M_E(i,j), & \text{if query } i \text{ is normal} \\
M_E(i,j), & \text{if query } i \text{ is extreme}
\end{cases}
$$

- `dozer_v1`: Normal queries attend to dozer mask keys while extreme queries attend to local window of Dozer mask and also the 
extreme keys.
$$
\
M_{ij} =
\begin{cases} 
M_{D}(i, j),  & \text{if query } i \text{ is normal} \\
M_E(i, j) \lor M_{local\_window}, & \text{if query } i \text{ is extreme}
\end{cases}
\
$$

