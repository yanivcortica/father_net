from dataset import SegDataset
import argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pathlib import Path
from leanai_core_yolo.segm_filter.utils import get_transform
from dataset import custom_collate, collate_pad
import torch
from leanai_core_yolo.segm_filter.utils import draw_semantic_segmentation_batch
import models
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

def main(cfg):
    model = getattr(models, cfg.model.name)(classes=cfg.num_classes, **cfg.model.args).to(cfg.device)
    model.load_state_dict(torch.load(cfg.draw.model, map_location=cfg.device))
    dataset = SegDataset(cfg.draw.dataset, transform=get_transform(cfg.test_transform))
    loader = DataLoader(dataset, batch_size=cfg.draw.batch_size, shuffle=True, collate_fn=custom_collate)
    samples = [next(iter(loader)) for _ in range(10)]
    batch = [(torch.squeeze(sam[0]).to(cfg.device),torch.squeeze(sam[1]).to(cfg.device)) for sam in samples]
    preds = [torch.squeeze(model(sam[0].to(cfg.device)).softmax(dim=1).argmax(dim=1)) for sam in samples]
    draw_semantic_segmentation_batch(batch,preds)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(Path(__file__).resolve().parent / args.cfg)
    main(cfg)