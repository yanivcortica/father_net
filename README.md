# Segmentation of crops

## Dangerous
- class indices for `SegDataset`
- read carefully `custom_collate` function, especially last resizing and reshaping part
- i try to configure everything via `.yaml` config. Proper way for experimenting is to create new config for each experiment.
- i assume model hasn't any activation on the last layer
- train/test split produced once and manually, no automation

## TODO
- repair eval and corresponding part of config
- repair tensorboard metrics

## List of experiments
- unet with pretrained resnet18
- hrnet with bigger batch size
- smaller lr
- `DiceLoss` and various coeffs for mixing two losses