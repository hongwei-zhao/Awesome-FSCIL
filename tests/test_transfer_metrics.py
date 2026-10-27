import random
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from transfer_metrics import (
    backward_transfer,
    compute_simplecil_baseline,
    forward_transfer,
    simplecil_backbone_name,
    task_accuracy_row,
)


class _ToyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, self.features[index], int(self.labels[index])


class _ToyDataManager:
    nb_tasks = 2
    nb_classes = 4

    def __init__(self):
        self.features = torch.eye(4, dtype=torch.float32)
        self.labels = np.arange(4)

    def get_task_size(self, task_id):
        return 2

    def get_dataset(self, indices, source, mode, kshot=None):
        del source, mode, kshot
        indices = np.asarray(indices)
        mask = np.isin(self.labels, indices)
        return _ToyDataset(self.features[mask], self.labels[mask])


class _IdentityBackbone(nn.Module):
    def forward(self, inputs):
        return inputs


class TransferMetricTests(unittest.TestCase):
    def test_metric_formulas(self):
        matrix = np.array(
            [
                [90.0, 80.0, 70.0],
                [88.0, 75.0, 65.0],
                [85.0, 72.0, 60.0],
            ]
        )
        self.assertEqual(backward_transfer(matrix), -4.0)
        self.assertEqual(forward_transfer(matrix, [85.0, 70.0, 50.0]), 12.5)

    def test_full_row_keeps_future_task_measurements(self):
        grouped = {
            "total": 50.0,
            "00-59": 80.0,
            "60-64": 70.0,
            "65-69": 60.0,
        }
        row = task_accuracy_row(grouped, 60, 5, 3)
        np.testing.assert_array_equal(row, [80.0, 70.0, 60.0])

    def test_missing_superdiagonal_is_rejected(self):
        matrix = np.array([[90.0, np.nan], [80.0, 70.0]])
        with self.assertRaisesRegex(ValueError, "R\[j-1,j\]"):
            forward_transfer(matrix, [90.0, 70.0])

    def test_method_backbone_names_map_to_plain_simplecil_backbones(self):
        self.assertEqual(
            simplecil_backbone_name("vit_base_patch16_224_coda_prompt"),
            "vit_base_patch16_224",
        )
        self.assertEqual(
            simplecil_backbone_name("pretrained_vit_b16_224_in21k_vpt"),
            "vit_base_patch16_224_in21k",
        )

    @mock.patch("transfer_metrics.timm.create_model")
    def test_dynamic_simplecil_baseline_uses_actual_model_outputs(self, create):
        create.return_value = _IdentityBackbone()
        args = {
            "backbone_type": "vit_base_patch16_224_l2p",
            "device": [torch.device("cpu")],
            "batch_size": 2,
            "num_workers": 0,
            "kshot": 1,
        }
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()
        expected_torch_state = torch.random.get_rng_state()

        result = compute_simplecil_baseline(args, _ToyDataManager())

        np.testing.assert_array_equal(result, [100.0, 100.0])
        create.assert_called_once_with(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        self.assertEqual(random.getstate(), expected_python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], expected_numpy_state[1])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_torch_state))


if __name__ == "__main__":
    unittest.main()
