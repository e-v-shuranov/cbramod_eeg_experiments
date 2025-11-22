import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator

import math

class Trainer(object):
    def __init__(self, params, data_loader, model):
        def cosine_warmup_mult(step):  # step: 0..warmup_steps-1
            if self.warmup_steps <= 0:
                return 1.0
            k = min(step, self.warmup_steps) / float(self.warmup_steps)
            # от start_factor к 1.0 по косинусу: 0 -> start_factor, 1 -> 1.0
            return self.start_factor + (1.0 - self.start_factor) * 0.5 * (1 - math.cos(math.pi * k))

        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])
        self.train_eval = Evaluator(params, self.data_loader['train'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        if self.params.use_scheduler:
            if self.params.use_cosine_warmup:
                self.total_steps = self.params.epochs * self.data_length
                self.warmup_steps = int(0.10 * self.total_steps)  # 10%
                self.decay_steps = max(1, self.total_steps - self.warmup_steps)  # 90%
                self.lr_max = self.params.lr
                self.lr_min = 1e-9
                self.start_factor = self.lr_min / self.lr_max
                warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=cosine_warmup_mult)
                decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.decay_steps, eta_min=self.lr_min
                )
                self.optimizer_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, decay],
                    milestones=[self.warmup_steps]  # после этого шага переключится на decay
                )
            else:
                self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-9
                )
        else:
            self.optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda _step: 1.0
            )

        print(self.model)

    def test_for_multiclass(self):
        print("model loading: " + self.params.model_for_test)
        map_location = torch.device(f'cuda:{self.params.cuda}')
        self.model.load_state_dict(torch.load(self.params.model_for_test, map_location=map_location,weights_only=True))


        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.train_eval.get_metrics_for_multiclass(self.model, store_embedings=self.params.store_embedings, path_emb_pkl=self.params.path_for_emb_storage_train)
            print("***************************Train results************************")
            print(
                "acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)

            acc_test, kappa_test, f1_test, cm_test = self.test_eval.get_metrics_for_multiclass(self.model, store_embedings=self.params.store_embedings, path_emb_pkl=self.params.path_for_emb_storage_test)
            print("***************************Test results************************")
            print(
                "acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc_test,
                    kappa_test,
                    f1_test,
                )
            )
            print(cm_test)

            acc_eval, kappa_eval, f1_eval, cm_eval = self.val_eval.get_metrics_for_multiclass(self.model, store_embedings=self.params.store_embedings, path_emb_pkl=self.params.path_for_emb_storage_val)
            print("***************************Val results************************")
            print(
                "acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc_eval,
                    kappa_eval,
                    f1_eval,
                )
            )

            print(cm_eval)

    def train_for_multiclass(self):
        if self.params.infer_only:
            self.test_for_multiclass()
            return
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y, f in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()


            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                acc_test, kappa_test, f1_test, cm_test = self.test_eval.get_metrics_for_multiclass(self.model)
                current_lr = optim_state['param_groups'][0]['lr']
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.2e}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        current_lr,
                        (timer() - start_time) / 60
                    )
                )
                print(
                    "Epoch {} : Testing_ Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.2e}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc_test,
                        kappa_test,
                        f1_test,
                        current_lr,
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                print(cm_test)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def test_for_binaryclass(self):
        print("model loading: " + self.params.model_for_test)
        map_location = torch.device(f'cuda:{self.params.cuda}')
        self.model.load_state_dict(torch.load(self.params.model_for_test, map_location=map_location,weights_only=True))

        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.train_eval.get_metrics_for_binaryclass(self.model, store_embedings=self.params.store_embedings, path_emb_pkl=self.params.path_for_emb_storage_train)


            print("***************************Train results************************")
            print(
                "acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)

            acc_test, pr_auc_test, roc_auc_test, cm_test = self.test_eval.get_metrics_for_binaryclass(self.model, store_embedings=self.params.store_embedings, path_emb_pkl=self.params.path_for_emb_storage_test)
            print("***************************Test results************************")
            print(
                "acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc_test,
                    pr_auc_test,
                    roc_auc_test,
                )
            )
            print(cm_test)

            acc_eval, pr_auc_eval, roc_auc_eval, cm_eval = self.val_eval.get_metrics_for_binaryclass(self.model, store_embedings=self.params.store_embedings, path_emb_pkl=self.params.path_for_emb_storage_val)
            print("***************************Val results************************")
            print(
                "acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc_eval,
                    pr_auc_eval,
                    roc_auc_eval,
                )
            )
            print(cm_eval)

    def train_for_binaryclass(self):
        if self.params.infer_only:
            self.test_for_binaryclass()
            return
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                acc_test, pr_auc_test, roc_auc_test, cm_test = self.test_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr']*10000000,
                        (timer() - start_time) / 60
                    )
                )
                print(
                    "Epoch {} : Testing_ Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc_test,
                        pr_auc_test,
                        roc_auc_test,
                        optim_state['param_groups'][0]['lr']*100000000,
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                print(cm_test)
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)


    def test_for_regression(self):
        print("model loading: " + self.params.model_for_test)
        map_location = torch.device(f'cuda:{self.params.cuda}')
        self.model.load_state_dict(torch.load(self.params.model_for_test, map_location=map_location,weights_only=True))

        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.train_eval.get_metrics_for_regression(self.model, store_embedings=self.params.store_embedings, path_emb_pkl=self.params.path_for_emb_storage_train)
            print("***************************Train results************************")
            print(
                "corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            corrcoef_test, r2_test, rmse_test = self.test_eval.get_metrics_for_regression(self.model, store_embedings=self.params.store_embedings, path_emb_pkl=self.params.path_for_emb_storage_test)
            print("***************************Test results************************")

            print(
                "corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef_test,
                    r2_test,
                    rmse_test,
                )
            )

            corrcoef_eval, r2_eval, rmse_eval = self.val_eval.get_metrics_for_regression(self.model, store_embedings=self.params.store_embedings, path_emb_pkl=self.params.path_for_emb_storage_val)
            print("***************************Val results************************")
            print(
                "corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef_eval,
                    r2_eval,
                    rmse_eval,
                )
            )


    def train_for_regression(self):
        if self.params.infer_only:
            self.test_for_regression()
            return
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr']*10000000,
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("r2 increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)