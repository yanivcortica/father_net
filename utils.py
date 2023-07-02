import os
import transforms
import albumentations
import torch
import numpy as np
import random
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from ignite.engine.engine import Engine
from ignite.engine import _check_arg, _prepare_batch
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
try:
    from torch.cuda.amp import autocast
except ImportError:
    raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")


def create_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable[[Any, Any, Any, torch.Tensor],
                               Any] = lambda x, y, y_pred, loss: loss.item(),
    amp_mode: Optional[str] = None,
    scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
    gradient_accumulation_steps: int = 1,
) -> Engine:

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    mode, _scaler = _check_arg(on_tpu, amp_mode, scaler)

    _update = train_step(
        model,
        optimizer,
        loss_fn,
        device,
        non_blocking,
        prepare_batch,
        output_transform,
        _scaler,
        gradient_accumulation_steps,
    )
    return Engine(_update)


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable[[Any, Any, Any, torch.Tensor],
                               Any] = lambda x, y, y_pred, loss: loss.item(),
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    gradient_accumulation_steps: int = 1,
) -> Callable:

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient_accumulation_steps must be strictly positive. "
            "No gradient accumulation if the value set to one (default)."
        )

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        with autocast(enabled=True):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if engine.state.iteration % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if engine.state.iteration % gradient_accumulation_steps == 0:
                optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return update





def read_img(im_path):
    return cv2.imread(im_path, 0)


def get_transform(cfg):
    L=[]
    impath = '/home/lean-ai-yaniv/data/401_imgs/images'
    t = albumentations.augmentations.domain_adaptation.HistogramMatching(reference_images=[os.path.join(
        impath, f) for f in os.listdir(impath)], p=1.0, read_fn=read_img, always_apply=True)
    L.append(t)
    L.extend([getattr(transforms, item.name)(**item.args) for item in cfg])

    return albumentations.Compose(L)


def setup_seeds(seed=42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_training_loss(engine):
    print(
        f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
    )
