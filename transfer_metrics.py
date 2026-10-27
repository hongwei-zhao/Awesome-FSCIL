"""Transfer metrics and the configuration-matched SimpleCIL baseline.

The project reports forward transfer relative to a frozen-pretrained-backbone
SimpleCIL classifier, not relative to a randomly initialized network.  The
baseline is therefore evaluated from the actual backbone and data split used
by an experiment instead of being stored as a dataset-only constant.
"""

from contextlib import contextmanager
import gc
import random

import numpy as np
import timm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


_METHOD_BACKBONE_SUFFIXES = (
    "_coda_prompt",
    "_dualprompt",
    "_adapter",
    "_inflora",
    "_sdlora",
    "_random",
    "_l2p",
    "_lae",
    "_vpt",
    "_ssf",
)


def simplecil_backbone_name(backbone_type):
    """Map a method-specific backbone name to its plain timm backbone."""
    name = str(backbone_type).lower()
    if name.startswith("pretrained_"):
        name = name[len("pretrained_") :]

    changed = True
    while changed:
        changed = False
        for suffix in _METHOD_BACKBONE_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break

    aliases = {
        "vit_b16_224": "vit_base_patch16_224",
        "vit_b16_224_in21k": "vit_base_patch16_224_in21k",
        "vit_b16_224_dino": "vit_base_patch16_224_dino",
        "vit_b16_224_sam": "vit_base_patch16_224_sam",
    }
    return aliases.get(name, name)


def task_accuracy_row(grouped, init_cls, increment, num_tasks):
    """Build one full task-accuracy row, preserving every evaluated task.

    Entries remain NaN only when the evaluator did not produce that task.  In
    particular, future-task entries must not be discarded merely because the
    corresponding task has not yet been trained.
    """
    row = np.full(num_tasks, np.nan, dtype=float)
    for task_id in range(num_tasks):
        if task_id == 0:
            start, end = 0, init_cls - 1
        else:
            start = init_cls + (task_id - 1) * increment
            end = start + increment - 1
        key = "{}-{}".format(str(start).rjust(2, "0"), str(end).rjust(2, "0"))
        if key in grouped:
            row[task_id] = grouped[key]
    return row


def _validated_square_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("accuracy matrix must be a square 2-D array")
    if matrix.shape[0] < 2:
        raise ValueError("transfer metrics require at least two tasks")
    return matrix


def backward_transfer(matrix):
    """Return mean R[T-1,j] - R[j,j] over all non-final tasks."""
    matrix = _validated_square_matrix(matrix)
    final_old = matrix[-1, :-1]
    learned = np.diag(matrix)[:-1]
    if not (np.isfinite(final_old).all() and np.isfinite(learned).all()):
        raise ValueError("BWT requires the final row and learned-task diagonal")
    return np.round(np.mean(final_old - learned), 2)


def forward_transfer(matrix, simplecil_acc):
    """Return FWT relative to a configuration-matched SimpleCIL baseline.

    ``simplecil_acc[j]`` is SimpleCIL's accuracy on task j immediately after
    adding that task.  This replaces the random-initialization baseline in the
    classical FWT definition with the stronger baseline used by this project.
    A scientifically comparable result additionally requires R[j-1,j] and the
    SimpleCIL run to use the same support set, transforms, classifier-building
    rule, and candidate-label space; those protocol choices live in the model
    evaluators and cannot be inferred from the accuracy matrix alone.
    """
    matrix = _validated_square_matrix(matrix)
    simplecil_acc = np.asarray(simplecil_acc, dtype=float)
    if simplecil_acc.ndim != 1 or simplecil_acc.shape[0] != matrix.shape[0]:
        raise ValueError(
            "SimpleCIL baseline length must equal the number of tasks"
        )

    before_learning = np.diag(matrix, k=1)
    baseline = simplecil_acc[1:]
    if not (np.isfinite(before_learning).all() and np.isfinite(baseline).all()):
        raise ValueError(
            "FWT requires every R[j-1,j] entry and SimpleCIL baseline value"
        )
    return np.round(np.mean(before_learning - baseline), 2)


def forward_transfer_or_unavailable(matrix, simplecil_acc):
    """Return FWT or an explicit explanation when future scores are absent."""
    try:
        return forward_transfer(matrix, simplecil_acc)
    except ValueError as error:
        return "N/A ({})".format(error)


def _resolve_device(args):
    devices = args.get("device", [-1])
    if not isinstance(devices, (list, tuple)):
        devices = [devices]
    device = devices[0]
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        if device.lower() == "cpu":
            return torch.device("cpu")
        if device.lower().startswith("cuda:"):
            return torch.device(device.lower())
        device = int(device)
    return torch.device("cpu" if int(device) < 0 else "cuda:{}".format(device))


@contextmanager
def _preserve_random_state(device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state(device)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def _extract_features(backbone, loader, device):
    features, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for _, inputs, targets in loader:
            output = backbone(inputs.to(device))
            if isinstance(output, dict):
                output = output.get("features", output.get("pre_logits"))
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output is None or output.ndim != 2:
                raise ValueError("SimpleCIL backbone did not return 2-D features")
            features.append(output.detach().cpu())
            labels.append(targets.detach().cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def compute_simplecil_baseline(args, data_manager):
    """Run the actual SimpleCIL prototype baseline for this configuration.

    The backbone is frozen.  At each session, prototypes are built from the
    same training split and k-shot policy as SimpleCIL, and the current task is
    classified against all prototypes observed up to that session.  These are
    the diagonal baseline values needed by this project's FWT definition.
    """
    device = _resolve_device(args)
    configured_name = args.get(
        "fwt_baseline_backbone_type", args["backbone_type"]
    )
    backbone_name = simplecil_backbone_name(configured_name)
    batch_size = int(args.get("batch_size", 64))
    num_workers = int(args.get("num_workers", 8))
    baseline = []

    with _preserve_random_state(device):
        backbone = timm.create_model(
            backbone_name,
            pretrained=bool(args.get("fwt_baseline_pretrained", True)),
            num_classes=0,
        ).to(device)
        backbone.eval()
        prototypes = []
        known_classes = 0

        try:
            for task_id in range(data_manager.nb_tasks):
                task_size = data_manager.get_task_size(task_id)
                total_classes = known_classes + task_size
                class_indices = np.arange(known_classes, total_classes)

                support_dataset = data_manager.get_dataset(
                    class_indices,
                    source="train",
                    mode="train",
                    kshot=args.get("kshot"),
                )
                support_loader = DataLoader(
                    support_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                )
                support_features, support_labels = _extract_features(
                    backbone, support_loader, device
                )
                for class_id in class_indices:
                    mask = support_labels == int(class_id)
                    if not mask.any():
                        raise ValueError(
                            "SimpleCIL baseline has no support for class {}".format(
                                class_id
                            )
                        )
                    prototypes.append(support_features[mask].mean(dim=0))

                query_dataset = data_manager.get_dataset(
                    class_indices, source="test", mode="test"
                )
                query_loader = DataLoader(
                    query_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
                weights = F.normalize(torch.stack(prototypes).to(device), dim=1)
                correct = 0
                total = 0
                backbone.eval()
                with torch.no_grad():
                    for _, inputs, targets in query_loader:
                        output = backbone(inputs.to(device))
                        if isinstance(output, dict):
                            output = output.get(
                                "features", output.get("pre_logits")
                            )
                        if isinstance(output, (tuple, list)):
                            output = output[0]
                        logits = F.normalize(output, dim=1) @ weights.t()
                        predictions = logits.argmax(dim=1).cpu()
                        correct += int((predictions == targets).sum().item())
                        total += int(targets.numel())
                if total == 0:
                    raise ValueError(
                        "SimpleCIL baseline has no test samples for task {}".format(
                            task_id
                        )
                    )
                baseline.append(round(100.0 * correct / total, 2))
                known_classes = total_classes
        finally:
            del backbone
            gc.collect()
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return np.asarray(baseline, dtype=float)
