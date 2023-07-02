import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def setup_data(cfg):
    mean = (86.45, 86.45, 86.45)
    std = (33.14, 33.14, 33.14)
    transform_train = A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensor(),
        ]
    )

    transform_val = A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensor(),
        ]
    )
    trainset = SegDataset(cfg.data_root, train=True, transform=transform_train)
    valset = SegDataset(cfg.data_root, train=False, transform=transform_val)
    train_loader = DataLoader(trainset, **cfg.train_dataloader)
    val_loader = DataLoader(valset, **cfg.val_dataloader)

    return train_loader, val_loader


# def setup_dataflow(dataset, train_args, test_args, train_idx=None, val_idx=None):
#     if all(v is not None for v in [train_idx, val_idx]):
#         train_sampler = SubsetRandomSampler(train_idx)
#         val_sampler = SubsetRandomSampler(val_idx)
#         train_loader = DataLoader(
#             dataset, **train_args, sampler=train_sampler, collate_fn=custom_collate
#         )
#         val_loader = DataLoader(
#             dataset, **test_args, sampler=val_sampler, collate_fn=custom_collate
#         )
#     else:
#         train_loader = DataLoader(
#             dataset, **train_args, shuffle=True, collate_fn=custom_collate
#         )
#         val_loader = DataLoader(dataset, **test_args, collate_fn=custom_collate)
#     return train_loader, val_loader


def init_semantic_segmentation_dataset(imgs_folder, masks_folder):
    names = os.listdir(imgs_folder)
    return [
        {
            "image": os.path.join(imgs_folder, name),
            "mask": os.path.join(masks_folder, f"{Path(name).stem}.png"),
        }
        for name in names
    ]


class SegDataset(Dataset):
    def __init__(
        self,
        data_root,
        train=True,
        transform=None,
    ):
        if train:
            img_dir = os.path.join(data_root, "img_dir", "train")
            ann_dir = os.path.join(data_root, "ann_dir", "train")
        else:
            img_dir = os.path.join(data_root, "img_dir", "val")
            ann_dir = os.path.join(data_root, "ann_dir", "val")
        self.dataset = init_semantic_segmentation_dataset(img_dir, ann_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {
            "image": cv2.imread(self.dataset[idx]["image"], cv2.IMREAD_GRAYSCALE),
            "mask": cv2.imread(self.dataset[idx]["mask"], cv2.IMREAD_GRAYSCALE),
        }

        if self.transform is not None:
            sample = self.transform(image=sample["image"], mask=sample["mask"])
            sample["mask"] = sample["mask"].long()
        return sample["image"], sample["mask"]
