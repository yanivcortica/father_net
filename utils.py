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


def find_class_of_seg(seg_img):
    '''get image of prediction or mask, and return the class based on the majority of pixels
    1 - pixels of crack
    2- pixels of watermarks

    :param seg_img: _description_
    :type seg_img: _type_
    '''
    seg_classes = [0, 0, 0]
    vls, cnts = torch.unique(seg_img, return_counts=True)
    for i, j in zip(vls, cnts):
        seg_classes[i] = j
    # we don't care about background (class 0)
    relevant_classes = seg_classes[1:]
    if sum(relevant_classes) == 0:
        return 0
    else:
        return int((torch.argmax(torch.Tensor(relevant_classes))) + 1)


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


def init_semantic_segmentation_dataset(data_path, imgs_folder, masks_folder):
    names = os.listdir(os.path.join(data_path, imgs_folder))
    return [
        {
            'image': os.path.join(data_path, imgs_folder, name),
            'mask': os.path.join(data_path, masks_folder, f'{Path(name).stem}_mask.png'),
        }
        for name in names
    ]


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


def draw_semantic_segmentation_batch(images, preds, count, masks=None):
    """ Draw batch from semantic segmentation dataset.

    Arguments:
        images (torch.Tensor): batch of images.
        masks_gt (torch.LongTensor): batch of ground-truth masks.
        plt (matplotlib.pyplot): canvas to show samples.
        masks_pred (torch.LongTensor, optional): batch of predicted masks.
        n_samples (int): number of samples to visualize.
    """
    ncols = 3 if masks else 2
    fig, ax = plt.subplots(nrows=len(images), ncols=ncols,
                           sharey=True, figsize=(10, 10))
    for i in range(len(images)):
        image = images[i].detach().cpu().numpy()
        pred = preds[i].detach().cpu().numpy()
        ax[i][0].imshow(image, cmap='gray')
        ax[i][0].set_xlabel("image")
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])

        ax[i][1].imshow(pred)
        ax[i][1].set_xlabel("predicted mask")
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])
        if masks:
            mask = masks[i].detach().cpu().numpy()
            ax[i][2].imshow(mask)
            ax[i][2].set_xlabel("ground truth")
            ax[i][2].set_xticks([])
            ax[i][2].set_yticks([])

    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.savefig(f'batch_seg_{count}.png')
    plt.close(fig)


def draw_semantic_segmentation(image, mask, pred, output_folder, count):
    """ Draw batch from semantic segmentation dataset.

    Arguments:
        images (torch.Tensor): batch of images.
        masks_gt (torch.LongTensor): batch of ground-truth masks.
        plt (matplotlib.pyplot): canvas to show samples.
        masks_pred (torch.LongTensor, optional): batch of predicted masks.
        n_samples (int): number of samples to visualize.
    """

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    image = image.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_xlabel("image")

    ax[1].imshow(pred, vmin=0, vmax=2)
    ax[1].set_xlabel("predicted mask")

    ax[2].imshow(mask, vmin=0, vmax=2)
    ax[2].set_xlabel("ground truth")

    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.savefig(os.path.join(output_folder, f'seg_{count}.png'))
    plt.close(fig)


def draw_iou(iou, label, output_folder):
    plt.plot(range(len(iou)), iou)
    plt.xlabel('epoch')
    plt.title(f'IoU-{label}')
    plt.savefig(os.path.join(output_folder, f'iou_{label}.png'))
    plt.close()


def plot_fp_fn(fp, fn, output_folder):
    plt.plot(range(1, len(fp)+1), fp, 'r', label='FP')
    plt.plot(range(1, len(fn)+1), fn, 'g', label='FN')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'tpr_tnr.png'))
    plt.close()
