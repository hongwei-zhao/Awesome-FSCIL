# Training and Inference Parameter Counts

This document distinguishes two quantities:

- **Train**: the maximum number of parameters that actually receive gradients in one FSCIL stage. Parameters stored in a tensor but detached or frozen are excluded.
- **Infer (functional stored delta)**: the parameters that must be retained in addition to one frozen backbone at the final stage. Auxiliary heads that are instantiated but never used in the forward result are excluded.

All values are parameter counts (not bytes) and are reported in millions (M, $10^6$). A mathematically mergeable LoRA delta can be folded into the backbone for deployment; that compact case is called out separately because it changes the backbone weights and is not the same as retaining an immutable backbone plus a delta.

## Symbols and repository settings

| Symbol | Meaning | ViT-B/16 value |
| --- | --- | ---: |
| $D$ | embedding width | 768 |
| $L$ | Transformer blocks | 12 |
| $r$ | LoRA rank | 10 |
| $C_0$, $C$, $c$ | base, final, and current incremental class counts | dataset-dependent |
| $T$ | number of stages | 11 for CUB/ImageNet-R; 9 for CIFAR/miniImageNet |

The repository uses either a biased linear head, $C(D+1)$, or a cosine head with one shared scale, $CD+1$. The formulas below follow the actual head used by each implementation.

## Audited formulas

| Method | Train | Infer (stored delta) | Audit result |
| --- | --- | --- | --- |
| SimpleFSCIL | 0 | $CD+1$ | Added. The frozen backbone only extracts features; cosine-classifier weights are class prototypes computed without gradient descent. |
| L2P | $DM(P+1)+C_0(D+1)$ | $P_B+DM(P+1)+C(D+1)$ | Corrected. $M=10$, $P=5$; the extra $DM$ is the learned key pool. The checked configs set `get_original_backbone=true`, so a second headless ViT ($P_B$) is required to produce the prompt-selection query. The instantiated but unused query-backbone classifier is excluded. |
| CODA-Prompt | $D\lfloor M/T\rfloor L_p(P+2)+C_0(D+1)$ | $DML_p(P+2)+C(D+1)$ | Corrected. Only the newly exposed component slice is trained per stage; the old document incorrectly counted the full pool for Train. Here $M=100$, $P=8$, $L_p=5$. |
| LAE-LoRA | $4DrL+C_0(D+1)$ | $8DrL+C(D+1)$ | Corrected. The online Q/V LoRA has $4DrL$ parameters; only the current class slice receives a classification gradient, and inference retains both the online and EMA experts. The old document used the wrong rank/effective count and omitted the EMA expert. |
| InfLoRA | $2DrL+C_0(D+1)$ | $4DrLT+C(D+1)$ | Train was correct: the paper fixes/designs $B_t$ and optimizes only $A_t$ (the repository names the optimized side `lora_B_*`). Infer was corrected: the task heads sum to $C$, not $TC_0$. The paper also permits folding every $A_tB_t$ into the backbone, reducing the added deployment parameters to $C(D+1)$. |
| SD-LoRA | $4DrL+1+C_0(D+1)$ | $4DrLT+T+C(D+1)$ | Corrected. The base stage is the maximum-gradient stage in these configurations: it has one magnitude scalar and the $C_0$ active classifier rows. A later stage has $t$ scalars but only the current $c$ classifier rows, so writing $T+C_0(D+1)$ incorrectly combines maxima from different stages. All Q/V directions can be algebraically folded into the backbone for a classifier-only added deployment. |
| ASP | $DPL+2f(P)+C_0D+1$ | $P_B+DPL+2f(P)+CD+1$ | Corrected. Train counts only the active base cosine-head rows; the preallocated future head does not receive the stage loss and is excluded. The prompt encoder calls a second frozen ViT backbone, so it must be counted at inference. $P_B=85{,}798{,}656$ for the headless ViT-B/16. |
| SEC-Prompt | $D(P+2)L_p(2N_0)+C_0D+1$ | $D(P+2)L_p(N_F+N_0)+CD+1$ | Corrected. Train uses the active base TSP and RSP pools plus the active base cosine head; it must not use the final classifier size. |
| SMP | $8DrL+2C_0D$ at the base stage; later stages train at most $CD$ classifier weights | $2D^2L+CD+1$ | Added. SMP trains two Q/V LoRA flows and two proxy-head weight matrices at the base stage (their scale parameters are frozen), then merges the flows into dense K/V deltas. Folding those deltas into QKV leaves only $CD+1$ added parameters. |

For ASP, with half prompt length $P$, each of the two Gaussian heads is

$$
f(P)=(2D\cdot256+256)+(256\cdot PD+PD).
$$

For SEC-Prompt in the checked configs, $P=3$, $L_p=4$, pool ratio $\rho=0.5$, $N_0=\operatorname{round}(\rho C_0)$, and $N_F=N_0+(T-1)\operatorname{round}(\rho c)$. The latter must not be simplified to $\operatorname{round}(\rho C)$: Python's rounding of $0.5\times5=2.5$ makes the CIFAR/miniImageNet final TSP pool 46 rather than 50. The base stage is the maximum because both TSP and RSP are active then; later stages train only the newly added TSP slice.

## Corrected numeric values

These values use the repository's ViT-B/16 IN1K configurations. `Infer` is the stored-delta representation, not the optional merged-backbone deployment.

| Method | CUB Train | CUB Infer | CIFAR Train | CIFAR Infer | miniImageNet Train | miniImageNet Infer | ImageNet-R Train | ImageNet-R Infer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleFSCIL | 0.00 | 0.15 | 0.00 | 0.08 | 0.00 | 0.08 | 0.00 | 0.15 |
| L2P | 0.12 | 86.00 | 0.09 | 85.92 | 0.09 | 85.92 | 0.12 | 86.00 |
| CODA-Prompt | 0.42 | 3.99 | 0.47 | 3.92 | 0.47 | 3.92 | 0.42 | 3.99 |
| LAE | 0.45 | 0.89 | 0.41 | 0.81 | 0.41 | 0.81 | 0.45 | 0.89 |
| InfLoRA | 0.26 | 4.21 | 0.23 | 3.39 | 0.23 | 3.39 | 0.26 | 4.21 |
| SD-LoRA | 0.45 | 4.21 | 0.41 | 3.39 | 0.41 | 3.39 | 0.45 | 4.21 |
| ASP | 2.08 | 87.95 | 2.04 | 87.87 | 2.04 | 87.87 | 4.90 | 90.78 |
| SEC-Prompt | 1.61 | 2.46 | 0.97 | 1.24 | 0.97 | 1.24 | 1.61 | 2.46 |
| SMP | 0.89 | 14.31 | 0.83 | 14.23 | 0.83 | 14.23 | 0.89 | 14.31 |

The merged-backbone Infer values for InfLoRA and SD-LoRA are 0.15 M on CUB/ImageNet-R and 0.08 M on CIFAR/miniImageNet. SMP has the same classifier-only merged values. ASP cannot remove its second backbone by LoRA merging, and LAE still needs two experts for its ensemble.

Full Finetune is not an additive-delta method: it updates the backbone itself. Its Train count is the full headless backbone plus the active classifier slice, $P_B+C_0(D+1)$, while its final standalone model contains $P_B+C(D+1)$ parameters (85.95 M for CUB/ImageNet-R and 85.88 M for CIFAR/miniImageNet). Treating only its classifier as an “added delta” would hide the fact that the backbone weights were replaced.

## Methods without a defensible paper-backed count

`TALON-MLP`, `TALON-QV`, the rank-sweep rows, `T0-KD`, `Uniform-KD`, `RCARD`, and `BC-FSCIL` are internal experiment labels. There is no corresponding original paper and no complete TALON implementation in this repository from which the old $2DrL$/$4DrL$ formulas could be reproduced. Their previous numeric claims have therefore been removed instead of being presented as verified.

## Correction history

| Revision | Methods with incorrect or unsupported counts | Resolution |
| --- | --- | --- |
| Original Chinese document | L2P; CODA-Prompt Train; LAE; InfLoRA Infer; SD-LoRA Train; ASP; SEC-Prompt Train; TALON-MLP/QV and the internal sweep/KD rows | The prompt/key pools, stage-specific freezing, online/EMA experts, task accumulation, second ASP backbone, active classifier rows, and Python rounding were re-derived from the papers and implementations. Unsupported internal-method numbers were removed. InfLoRA Train and the previously reported CODA-Prompt Infer structure were already correct. |
| Previous English revision | ASP Train; the exact SD-LoRA Train formula | ASP Train incorrectly used the final $C$ rows instead of the active $C_0$ rows. SD-LoRA incorrectly combined the final number of magnitude scalars with the base-stage classifier, although this did not change the displayed two-decimal values. Both are corrected above. |
| Current revision | None known under the definitions at the top of this document | Every listed count now follows the same maximum-gradient-stage and functional-stored-delta conventions; optional algebraic merging is reported separately. |

## Primary sources and implementation evidence

- L2P: [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654), Section 5.1 ($M=10$, $P=5$, 46,080 prompt/key parameters).
- CODA-Prompt: [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.html), Sections 4.2–4.3 and appendix ($P$, $K$, $A$ components; old components frozen).
- LAE: [A Unified Continual Learning Framework with General Parameter-Efficient Tuning](https://arxiv.org/abs/2303.10070), Sections 3.2–3.3 and Table XI (online/EMA experts and ensemble inference).
- InfLoRA: [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html), Section 3 ($B_t$ designed/frozen, $A_t$ optimized, branches mergeable).
- SD-LoRA: [ICLR 2025 paper](https://arxiv.org/abs/2501.13198), Eq. 4 and Section 4 (Q/V directions and learned magnitudes).
- ASP: [ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10470.pdf), Section 4.2 (the prompt encoder uses a pre-trained backbone).
- SEC-Prompt: [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_SEC-PromptSEmantic_Complementary_Prompting_for_Few-Shot_Class-Incremental_Learning_CVPR_2025_paper.html) and the checked `DPrompt`/`NDPrompt` implementation.
- SMP: [IEEE TCSVT paper](https://ieeexplore.ieee.org/document/11422019), [preprint](https://arxiv.org/abs/2508.05094), and the checked two-flow adapter-merging implementation in `SMP-main`.

Repository-specific counts were cross-checked against the tensor shapes, detach/freeze rules, optimizer parameter lists, and logged parameter totals. A raw `requires_grad` count taken before a learner applies its stage-specific freezing is not treated as the actual Train count.
