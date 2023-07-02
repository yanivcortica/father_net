from pathlib import Path

from dataset import setup_data
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from utils import setup_seeds
from omegaconf import OmegaConf
from ignite.contrib.handlers import ProgressBar
import torch
from torch.utils.data import DataLoader
import torchmetrics as tm
import torch.optim as optim
import segmentation_models_pytorch as smp
from ignite.metrics import JaccardIndex, Loss
import argparse
import models
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix


class Experiment:
    def __init__(self, cfg):
        a = 1

        setup_seeds()
        self.cfg = cfg
        self.dataloader_train, self.dataloader_val = setup_data()
        self.device = torch.device(cfg.device)
        self.model = getattr(models, cfg.model.name)(**cfg.model.args).to(self.device)
        self.optimizer = getattr(optim, cfg.optimizer.type)(
            params=self.model.parameters(), **cfg.optimizer.args
        )
        self.loss_fn = getattr(smp.losses, cfg.loss.name)(**cfg.loss.args)
        
        # define metrics
        self.metric_fn = getattr(tm, cfg.metrics.name)(**cfg.metrics.args)

    def init(self):
        # for k,v in model.named_parameters():
        #     if k.startswith('encoder'):
        #         v.requires_grad = False

        self.loss_fn = self.loss_fn.to(self.device)

        trainer = create_trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
            **self.cfg.train.init,
        )

        evaluator = create_supervised_evaluator(
            model=model,
            metrics={
                "Loss": Loss(self.loss_fn),
                "Jaccard": JaccardIndex(cm),
            },
            device=self.device,
        )
        return model, trainer, evaluator

    def train(self):
        # trainloader, testloader = setup_dataflow(self.trainset,cfg.train_dataloader,cfg.test_dataloader,None,None)
        trainloader = DataLoader(
            self.trainset, **cfg.train_dataloader, collate_fn=custom_collate
        )
        testloader = DataLoader(
            self.testset, **cfg.test_dataloader, collate_fn=custom_collate
        )
        model, trainer, evaluator = self.init()
        train_results, val_results, grads = self.train_model(
            model, trainer, evaluator, trainloader, testloader, save=True
        )
        return train_results, val_results, grads

    def train_model(
        self, model, trainer, evaluator, trainloader, testloader, save=False
    ):
        train_results = []
        val_results = []
        grads = []
        all_fp = []
        all_fn = []

        if save:
            ckpt_saver = ModelCheckpoint(
                cfg.checkpoints_folder,
                f"{cfg.experiment_name}",
                n_saved=None,
                create_dir=True,
                require_empty=False,
            )

            trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=1), ckpt_saver, {"ckpt": model}
            )

        # @trainer.on(Events.EPOCH_COMPLETED(once=2))
        def unfreeze_encoder():
            for k, v in model.named_parameters():
                if k.startswith("encoder"):
                    v.requires_grad = True

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_test_results():
            evaluator.run(testloader)
            metrics = evaluator.state.metrics.copy()
            val_results.append(metrics)
            s = f"Validation Results - Epoch[{trainer.state.epoch}]\ncrack-IoU: {metrics['Jaccard'][1]:.2f}\nwatermark-IoU: {metrics['Jaccard'][2]:.2f} \nAvg loss: {metrics['Loss']:.2f}"
            print(s)

        @trainer.on(Events.EPOCH_COMPLETED)
        def cm():
            y_true = []
            y_pred = []
            for i, x in enumerate(testloader):
                pred = torch.squeeze(
                    model(x[0].to(cfg.device)).softmax(dim=1).argmax(dim=1)
                )
                mask = x[1].squeeze()
                pred_class = find_class_of_seg(pred)
                mask_class = find_class_of_seg(mask)
                if pred_class == 0 or mask_class == 0:
                    continue
                y_true.append(0 if mask_class == 2 else mask_class)
                y_pred.append(0 if pred_class == 2 else pred_class)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            all_fp.append(fp)
            all_fn.append(fn)

        # @trainer.on(Events.ITERATION_COMPLETED)
        # def log_grad_norm():
        #     total_norm = 0
        #     for p in model.parameters():
        #         param_norm = p.grad.detach().data.norm(2)
        #         total_norm += param_norm.item() ** 2
        #     total_norm = total_norm ** 0.5
        #     grads.append(total_norm)

        @trainer.on(Events.EPOCH_COMPLETED(every=self.cfg.train.run.max_epochs))
        def draw_validation():
            (
                save_folder,
                TP_folder,
                FP_folder,
                TN_folder,
                FN_folder,
            ) = self._create_folders()
            for i, x in enumerate(testloader):
                pred = torch.squeeze(
                    model(x[0].to(cfg.device)).softmax(dim=1).argmax(dim=1)
                )
                pred_class = find_class_of_seg(pred)
                mask_class = find_class_of_seg(torch.squeeze(x[1]))
                prediction_status = self._get_prediction_status(pred_class, mask_class)
                draw_folder = self._get_draw_folder(
                    prediction_status, TP_folder, FP_folder, TN_folder, FN_folder
                )
                draw_semantic_segmentation(
                    torch.squeeze(x[0]), torch.squeeze(x[1]), pred, draw_folder, i
                )
            val_loss = [float(val["Loss"]) for val in val_results]
            train_loss = [float(train["Loss"]) for train in train_results]
            cracks = [float(val["Jaccard"][1]) for val in val_results]
            wm = [float(val["Jaccard"][2]) for val in val_results]
            a = 1
            draw_iou(cracks, "cracks", save_folder)
            draw_iou(wm, "watermarks", save_folder)
            draw_iou(val_loss, "validation_loss", save_folder)
            draw_iou(train_loss, "train_loss", save_folder)
            plot_fp_fn(all_fp, all_fn, save_folder)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results():
            evaluator.run(trainloader)
            metrics = evaluator.state.metrics.copy()
            train_results.append(metrics)
            s = f"Training Results - Epoch[{trainer.state.epoch}]\nbg-IoU: {metrics['Jaccard'][0]:.2f}\ncrack-IoU: {metrics['Jaccard'][1]:.2f}\nwatermark-IoU: {metrics['Jaccard'][2]:.2f} \nAvg loss: {metrics['Loss']:.2f}"
            # print(s)

        ProgressBar().attach(trainer, output_transform=lambda x: {"batch loss": x})
        trainer.run(trainloader, **self.cfg.train.run)

        return train_results, val_results, grads


def main(cfg):
    experiment = Experiment(cfg)
    experiment.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(Path(__file__).resolve().parent / args.cfg)
    main(cfg)
