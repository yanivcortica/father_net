import random

import cv2
import torch
import torchvision.transforms.functional as TF
from leanai_core_yolo.segm_filter.utils import init_semantic_segmentation_dataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from albumentations.augmentations.geometric.rotate import RandomRotate90


def setup_dataflow(dataset,train_args,test_args, train_idx=None, val_idx=None):
    if all(v is not None for v in [train_idx,val_idx]):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, **train_args, sampler=train_sampler,collate_fn=custom_collate)
        val_loader = DataLoader(dataset, **test_args, sampler=val_sampler, collate_fn=custom_collate)
    else:
        train_loader = DataLoader(dataset, **train_args,shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(dataset, **test_args, collate_fn=custom_collate)
    return train_loader, val_loader

class SegDataset(Dataset):
    def __init__(
        self,
        root,
        imgs_folder="images",
        masks_folder="masks",
        num_classes=3,
        transform=None,
        class_names=None,
    ):
        # self.class_idxs = class_idxs
        self.num_classes = num_classes
        self.dataset = init_semantic_segmentation_dataset(
            root, imgs_folder, masks_folder
        )
        self.transform = transform
        self.class_names = class_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # im_path, mask_path = self.paths[idx]
        # im = cv2.imread(im_path)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[:, :, None]
        # for i, class_idx in enumerate(self.class_idxs):
        #     mask[mask == class_idx] = i + 1

        # if mask is None:
        #     print(mask_path)

        # transformed = self.transform(image=im, mask=mask)
        # return transformed['image'], transformed['mask'].float()
        #print(self.dataset[idx]["mask"])
        sample = {
            "image": cv2.imread(self.dataset[idx]["image"],0),
            "mask": cv2.imread(self.dataset[idx]["mask"], 0),
        }
        
        if self.transform is not None:

            sample = self.transform(image=sample['image'],mask=sample['mask'])
            sample["mask"] = sample["mask"].long()
        return sample['image'],sample['mask']


def collate_pad(batch):
    shapes = [list(img.shape[1:]) for img, mask in batch]
    H, W = torch.tensor(shapes).max(dim=0)[0].tolist()
    H = ((H + 31) // 32) * 32
    W = ((W + 31) // 32) * 32

    #print(H, W)

    padded_imgs = []
    padded_masks = []
    for (h, w), (img, mask) in zip(shapes, batch):
        h1 = random.randint(0, H - h)
        h2 = H - h - h1
        w1 = random.randint(0, W - w)
        w2 = W - w - w1
        padding = (w1, h1, w2, h2)
        padded = TF.pad(img, padding)
        padded_imgs.append(padded)
        _, hh, ww = padded.shape

        padded_masks.append(TF.pad(mask, padding))
        # TODO: only for hrnet divide by 2
        #padded_masks.append(TF.resize(TF.pad(mask, padding), (hh // 2, ww // 2)))
    return torch.stack(padded_imgs, dim=0), torch.stack(padded_masks, dim=0)

def rotate_and_align_batch(batch):
    if len(batch)==1:
        return batch
    k = random.choice([1,3])
    im1 = batch[0][0][0,:,:]
    mask1 = batch[0][1]
    im2 = batch[1][0][0,:,:]
    mask2 = batch[1][1]
    if random.random()<0.5:
        im1 = torch.rot90(im1,k)
        mask1 = torch.rot90(mask1,k)
    ratio1 = im1.shape[0]/im1.shape[1]
    ratio2 = im2.shape[0]/im2.shape[1]
    need_rotation = (ratio1<=1 and ratio2>1) or (ratio1>=1 and ratio2<1)
    if need_rotation:
        im2 = torch.rot90(im2,k)
        mask2 = torch.rot90(mask2,k)
    return [(im1[None,:],mask1),(im2[None,:],mask2)]

def custom_collate(batch):

    #batch = rotate_and_align_batch(batch)

    shapes = [list(img.shape[1:]) for img, mask in batch]
    H, W = torch.tensor(shapes).max(dim=0)[0].tolist()
    H = ((H + 31) // 32) * 32
    W = ((W + 31) // 32) * 32

    #print(H, W)

    padded_imgs = []
    padded_masks = []
    for (h, w), (img, mask) in zip(shapes, batch):
        h1 = random.randint(0, H - h)
        h2 = H - h - h1
        w1 = random.randint(0, W - w)
        w2 = W - w - w1
        padding = (w1, h1, w2, h2)
        padded = TF.pad(img, padding)
        padded_imgs.append(padded)
        _, hh, ww = padded.shape

        padded_masks.append(TF.pad(mask, padding))
        # TODO: only for hrnet divide by 2
        #padded_masks.append(TF.resize(TF.pad(mask, padding), (hh // 2, ww // 2)))
    return torch.stack(padded_imgs, dim=0), torch.stack(padded_masks, dim=0)
