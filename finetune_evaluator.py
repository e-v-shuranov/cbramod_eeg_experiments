import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error
from tqdm import tqdm

import pickle

class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_metrics_for_multiclass(self, model, store_embedings=False, path_emb_pkl=""):
        model.eval()
        if store_embedings:
            signal_for_store = [[],[],[]]
            emb_for_store = []
            target_for_store = []


        truths = []
        preds = []
        for x, y, file_names in tqdm(self.data_loader, total=len(self.data_loader), mininterval=1):
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()

            if store_embedings:
                signal_for_store[0].extend(file_names)
                # emb_for_store.extend([[pred.cpu().numpy().reshape(64,-1),emb.cpu().numpy().reshape(64,-1)]])
                emb_for_store.extend([pred.cpu().numpy()])
                target_for_store.extend(y.cpu().numpy())

        if store_embedings:
            with open(path_emb_pkl, 'wb') as handle:
                pickle.dump([signal_for_store, emb_for_store, target_for_store],
                            handle)  # , protocol=pickle.HIGHEST_PROTOCOL)
        truths = np.array(truths)
        preds = np.array(preds)
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model, store_embedings=False, path_emb_pkl=""):
        model.eval()
        if store_embedings:
            signal_for_store = [[],[],[]]
            emb_for_store = []
            target_for_store = []
        truths = []
        preds = []
        scores = []
        for x, y, file_names in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()
            if store_embedings:
                signal_for_store[0].extend(file_names)
                # emb_for_store.extend([[pred.cpu().numpy().reshape(64,-1),emb.cpu().numpy().reshape(64,-1)]])
                emb_for_store.extend([pred.cpu().numpy()])
                target_for_store.extend(y.cpu().numpy())

        if store_embedings:
            with open(path_emb_pkl, 'wb') as handle:
                pickle.dump([signal_for_store, emb_for_store, target_for_store],
                            handle)  # , protocol=pickle.HIGHEST_PROTOCOL)
        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(self, model, store_embedings=False, path_emb_pkl=""):
        model.eval()
        if store_embedings:
            signal_for_store = [[],[],[]]
            emb_for_store = []
            target_for_store = []
        truths = []
        preds = []
        for x, y, file_names in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred.cpu().squeeze().numpy().tolist()
            if store_embedings:
                signal_for_store[0].extend(file_names)
                # emb_for_store.extend([[pred.cpu().numpy().reshape(64,-1),emb.cpu().numpy().reshape(64,-1)]])
                emb_for_store.extend([pred.cpu().numpy()])
                target_for_store.extend(y.cpu().numpy())

        if store_embedings:
            with open(path_emb_pkl, 'wb') as handle:
                pickle.dump([signal_for_store, emb_for_store, target_for_store],
                            handle)  # , protocol=pickle.HIGHEST_PROTOCOL)
        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse