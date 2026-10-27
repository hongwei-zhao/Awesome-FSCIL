# SMP experiment status

Publication: **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2026** ([IEEE Xplore](https://ieeexplore.ieee.org/document/11422019)).

Statistics date: 2026-08-12. Each setting contains four datasets
(`CUB200`, `CIFAR100`, `ImageNet-R`, and `miniImageNet`), for a total of
16 planned runs.

| Setting | Completed | Status |
| --- | ---: | --- |
| `IN1K-NoShuffle1993` | 4 / 4 | Complete |
| `IN1K-Shuffle1993` | 4 / 4 | Complete |
| `IN1K-Shuffle2025` | 4 / 4 | Complete |
| `IN1K-Shuffle42` | 4 / 4 | Complete |
| **Total** | **16 / 16 (100%)** | **Complete** |

## Completed metrics

Values below are taken from the final `Top1 curve` in the retained SMP output
logs. `Base`, `Last`, and `Avg` are recorded in
[`../FSCIL_IN1K_Baselines.xlsx`](../FSCIL_IN1K_Baselines.xlsx); `PD` is
calculated in the workbook as `Base - Last`.

| Setting | Dataset | Base | Last | Avg | Top-1 curve |
| --- | --- | ---: | ---: | ---: | --- |
| `IN1K-NoShuffle1993` | CUB200 | 89.33 | 86.17 | 86.78 | 89.33, 88.74, 87.87, 86.98, 86.73, 85.71, 85.58, 85.94, 85.47, 86.04, 86.17 |
| `IN1K-NoShuffle1993` | CIFAR100 | 94.20 | 88.45 | 90.87 | 94.20, 92.63, 92.04, 90.61, 90.70, 90.01, 89.88, 89.33, 88.45 |
| `IN1K-NoShuffle1993` | ImageNet-R | 87.63 | 76.38 | 80.75 | 87.63, 84.89, 84.18, 81.80, 80.71, 80.18, 78.94, 78.22, 77.94, 77.38, 76.38 |
| `IN1K-NoShuffle1993` | miniImageNet | 98.23 | 95.71 | 96.72 | 98.23, 97.98, 96.91, 96.76, 96.81, 96.36, 95.87, 95.82, 95.71 |
| `IN1K-Shuffle1993` | CUB200 | 93.01 | 84.86 | 88.50 | 93.01, 92.33, 90.18, 89.17, 88.54, 87.95, 88.00, 87.06, 86.56, 85.84, 84.86 |
| `IN1K-Shuffle1993` | CIFAR100 | 95.28 | 88.45 | 91.50 | 95.28, 94.11, 94.01, 92.27, 90.61, 90.06, 89.39, 89.34, 88.45 |
| `IN1K-Shuffle1993` | ImageNet-R | 87.30 | 77.00 | 81.48 | 87.30, 85.78, 84.33, 82.81, 82.23, 81.16, 80.13, 78.91, 78.93, 77.69, 77.00 |
| `IN1K-Shuffle1993` | miniImageNet | 97.83 | 94.92 | 96.13 | 97.83, 97.48, 97.30, 95.99, 95.81, 95.52, 95.47, 94.86, 94.92 |
| `IN1K-Shuffle2025` | CUB200 | 93.03 | 85.88 | 89.20 | 93.03, 92.57, 90.88, 90.33, 89.98, 88.43, 88.27, 87.70, 87.68, 86.40, 85.88 |
| `IN1K-Shuffle2025` | CIFAR100 | 95.55 | 88.46 | 91.62 | 95.55, 94.49, 93.30, 92.56, 91.32, 90.61, 89.32, 88.94, 88.46 |
| `IN1K-Shuffle2025` | ImageNet-R | 87.76 | 77.83 | 81.58 | 87.76, 84.36, 83.66, 83.21, 82.31, 81.76, 80.32, 79.52, 78.60, 78.01, 77.83 |
| `IN1K-Shuffle2025` | miniImageNet | 97.92 | 95.26 | 96.14 | 97.92, 96.69, 96.59, 95.96, 95.98, 96.00, 95.60, 95.28, 95.26 |
| `IN1K-Shuffle42` | CUB200 | 92.92 | 86.34 | 88.63 | 92.92, 89.99, 89.32, 89.01, 89.70, 88.86, 88.59, 86.76, 86.64, 86.77, 86.34 |
| `IN1K-Shuffle42` | CIFAR100 | 94.82 | 88.98 | 91.91 | 94.82, 94.23, 93.41, 92.79, 92.06, 91.19, 90.16, 89.52, 88.98 |
| `IN1K-Shuffle42` | ImageNet-R | 88.50 | 79.93 | 83.60 | 88.50, 86.99, 86.17, 84.94, 83.54, 83.00, 82.76, 81.80, 81.39, 80.55, 79.93 |
| `IN1K-Shuffle42` | miniImageNet | 97.87 | 95.49 | 96.54 | 97.87, 97.66, 97.46, 97.23, 96.15, 95.73, 95.69, 95.56, 95.49 |

The runner does not report BWT or FWT, so those workbook fields are left
blank rather than inferred from aggregate accuracy.
