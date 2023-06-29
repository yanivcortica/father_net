from pathlib import Path

from dataset import SegDataset, setup_dataflow
from leanai_core_yolo.segm_filter.utils import (
    get_transform,
    setup_seeds,
)
from leanai_core_yolo.segm_filter.utils import draw_semantic_segmentation, draw_iou, create_trainer, find_class_of_seg, plot_fp_fn
from leanai_core_yolo.segm_filter.dataset import custom_collate

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint

from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from ignite.contrib.handlers import ProgressBar
import torch
from torch.utils.data import DataLoader
import torchmetrics as tm
import torch.optim as optim
from ignite.metrics import JaccardIndex, Loss
import argparse
from sklearn.model_selection import KFold
import models
from ignite.metrics import ConfusionMatrix
import statistics
from sklearn.metrics import confusion_matrix


class Experiment:
    def __init__(self, cfg):

        self.cfg = cfg

        self.trainset = SegDataset(
            **cfg.train_ds, transform=get_transform(cfg.train_transform)
        )
        self.testset = SegDataset(
            **cfg.test_ds, transform=get_transform(cfg.test_transform)
        )
        
        self.device = torch.device(cfg.device)
        # define loss
        # self.loss_fn = getattr(smp.losses, cfg.loss.name)(**cfg.loss.args)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # define metrics
        self.metric_fn = getattr(tm, cfg.metrics.name)(**cfg.metrics.args)
        
        setup_seeds()
        
    def init(self):
        cm = ConfusionMatrix(self.cfg.num_classes)
        
        model = getattr(models, cfg.model.name)(**cfg.model.args).to(self.device)
        
        # for k,v in model.named_parameters():
        #     if k.startswith('encoder'):
        #         v.requires_grad = False
                
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay)
        
        self.loss_fn = self.loss_fn.to(self.device)
        
        trainer = create_trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
            **self.cfg.train.init,)

        evaluator = create_supervised_evaluator(
            model=model,
            metrics={
                "Loss": Loss(self.loss_fn),
                "Jaccard": JaccardIndex(cm),
            },
            device=self.device)
        return model, trainer, evaluator
    
    def train_kfold(self):
        
        kfold = KFold(n_splits=self.cfg.kfold.num_folds, shuffle=True)
        
        train_res = {i+1: [] for i in range(kfold.n_splits)}
        val_res = {i+1: [] for i in range(kfold.n_splits)}

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.trainset)):
            
            print(f"Fold {fold_idx + 1}")
            print('------------------------')
            train_loader, val_loader = setup_dataflow(self.trainset,cfg.train_dataloader,cfg.test_dataloader,train_idx,val_idx)
            model,trainer,evaluator = self.init()
            train_results, val_results = self.train_model(model, trainer, evaluator, train_loader, val_loader)
            train_res[fold_idx+1].extend(train_results)
            val_res[fold_idx+1].extend(val_results)

        return train_res, val_res

    def train(self):
        
        #trainloader, testloader = setup_dataflow(self.trainset,cfg.train_dataloader,cfg.test_dataloader,None,None)
        trainloader = DataLoader(self.trainset, **cfg.train_dataloader,collate_fn=custom_collate)
        testloader = DataLoader(self.testset, **cfg.test_dataloader,collate_fn=custom_collate)
        model,trainer,evaluator = self.init()
        train_results, val_results, grads = self.train_model(model, trainer, evaluator, trainloader, testloader, save=True)
        return train_results, val_results, grads
    
    def train_model(self, model, trainer, evaluator, trainloader, testloader, save=False):
        train_results = []
        val_results = []
        grads = []
        all_fp = []
        all_fn = []
        
        if save:
            ckpt_saver = ModelCheckpoint(
                cfg.checkpoints_folder,
                f'{cfg.experiment_name}',
                n_saved=None,
                create_dir=True,
                require_empty=False)
            
            trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_saver, {"ckpt": model})
        
        #@trainer.on(Events.EPOCH_COMPLETED(once=2))
        def unfreeze_encoder():
            for k,v in model.named_parameters():
                if k.startswith('encoder'):
                    v.requires_grad = True
                    
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_test_results():
            evaluator.run(testloader)
            metrics = evaluator.state.metrics.copy()
            val_results.append(metrics)
            s = (
                f"Validation Results - Epoch[{trainer.state.epoch}]\ncrack-IoU: {metrics['Jaccard'][1]:.2f}\nwatermark-IoU: {metrics['Jaccard'][2]:.2f} \nAvg loss: {metrics['Loss']:.2f}")
            print(s)
        
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def cm():
            y_true = []
            y_pred = []
            for i,x in enumerate(testloader):
                pred = torch.squeeze(model(x[0].to(cfg.device)).softmax(dim=1).argmax(dim=1))
                mask = x[1].squeeze()
                pred_class = find_class_of_seg(pred)
                mask_class = find_class_of_seg(mask)
                if pred_class==0 or mask_class==0:
                    continue
                y_true.append(0 if mask_class==2 else mask_class)
                y_pred.append(0 if pred_class==2 else pred_class)
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
            save_folder,TP_folder,FP_folder,TN_folder,FN_folder = self._create_folders()
            for i,x in enumerate(testloader):
                pred = torch.squeeze(model(x[0].to(cfg.device)).softmax(dim=1).argmax(dim=1))
                pred_class = find_class_of_seg(pred)
                mask_class = find_class_of_seg(torch.squeeze(x[1]))
                prediction_status = self._get_prediction_status(pred_class,mask_class)
                draw_folder = self._get_draw_folder(prediction_status,TP_folder,FP_folder,TN_folder,FN_folder)
                draw_semantic_segmentation(torch.squeeze(x[0]), torch.squeeze(x[1]), pred, draw_folder, i)
            val_loss = [float(val['Loss']) for val in val_results]
            train_loss = [float(train['Loss']) for train in train_results]
            cracks = [float(val['Jaccard'][1]) for val in val_results]
            wm = [float(val['Jaccard'][2]) for val in val_results]
            a=1
            draw_iou(cracks, 'cracks', save_folder)
            draw_iou(wm, 'watermarks', save_folder)
            draw_iou(val_loss, 'validation_loss', save_folder)
            draw_iou(train_loss, 'train_loss',save_folder)
            plot_fp_fn(all_fp, all_fn, save_folder)
            
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results():
            evaluator.run(trainloader)
            metrics = evaluator.state.metrics.copy()
            train_results.append(metrics)
            s = (f"Training Results - Epoch[{trainer.state.epoch}]\nbg-IoU: {metrics['Jaccard'][0]:.2f}\ncrack-IoU: {metrics['Jaccard'][1]:.2f}\nwatermark-IoU: {metrics['Jaccard'][2]:.2f} \nAvg loss: {metrics['Loss']:.2f}")
            #print(s)


        
        ProgressBar().attach(trainer, output_transform=lambda x: {"batch loss": x})
        trainer.run(trainloader, **self.cfg.train.run)

        return train_results, val_results, grads
    
    def _get_prediction_status(self,pred_class,mask_class):
        if mask_class == 1:
            return 'TP' if pred_class == 1 else 'FN'
        elif mask_class in [0,2]:
            return 'TN' if pred_class in [0,2] else 'FP'
            
    def _create_folders(self):
        save_folder = Path(f'{self.cfg.results_folder}_{self.cfg.experiment_name}')
        save_folder.mkdir(parents=True,exist_ok=True)
        TP_folder = Path(f'{self.cfg.results_folder}_{self.cfg.experiment_name}') / 'TP'
        FP_folder = Path(f'{self.cfg.results_folder}_{self.cfg.experiment_name}') / 'FP'
        TN_folder = Path(f'{self.cfg.results_folder}_{self.cfg.experiment_name}') / 'TN'
        FN_folder = Path(f'{self.cfg.results_folder}_{self.cfg.experiment_name}') / 'FN'
        TP_folder.mkdir(parents=True,exist_ok=True)
        FP_folder.mkdir(parents=True,exist_ok=True)
        TN_folder.mkdir(parents=True,exist_ok=True)
        FN_folder.mkdir(parents=True,exist_ok=True)
        return str(save_folder),str(TP_folder),str(FP_folder),str(TN_folder),str(FN_folder)
    
    def _get_draw_folder(self,prediction_status,TP_folder,FP_folder,TN_folder,FN_folder):
        if prediction_status == 'TP':
            return TP_folder
        if prediction_status == 'FP':
            return FP_folder
        if prediction_status == 'TN':
            return TN_folder
        if prediction_status == 'FN':
            return FN_folder
        
# train_transform = A.Compose(
#     [
#         # A.Resize(224, 224),
#         #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
#         A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(transpose_mask=True),
#     ]

def get_k_fold_results(cfg, train_results, validation_results):
    cracks_iou = []
    wm_iou = []
    for k in range(cfg.train.run.max_epochs):
        cracks_iou.append(statistics.fmean([validation_results[i][k]['Jaccard'][1] for i in range(1,cfg.kfold.num_folds+1)]))
        wm_iou.append(statistics.fmean([validation_results[i][k]['Jaccard'][2] for i in range(1,cfg.kfold.num_folds+1)]))
    return cracks_iou, wm_iou

def main(cfg):

    experiment = Experiment(cfg)
    if cfg.kfold.train_kfold:
        train_results, validation_results = experiment.train_kfold()
        cracks_iou, wm_iou = get_k_fold_results(cfg, train_results, validation_results)
        a=1
    else:
        train_results, validation_results, grads = experiment.train()
        a=1
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(Path(__file__).resolve().parent / args.cfg)
    main(cfg)
