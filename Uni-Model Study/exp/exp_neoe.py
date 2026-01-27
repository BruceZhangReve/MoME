from data.data_loader import Dataset_MTS
from exp.exp_basic import Exp_Basic
from models.patchtst import PatchTSTC
from models.tsmixer import TSMixerC
from models.Dlinear import DLinearC
from models.timesnet import TimesNetC
from models.tmoe_old import TMoE
from models.tmoe_v3 import TMoE_v3
from models.mlp import MLP
from models.iTransformer import Model as Itransformer
import torch.nn.functional as F
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, MSE_dim

import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import wandb
import os
import time
import json
import pickle
from torchinfo import summary

#For Ploting Purpose
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import math


class Exp_neoe(Exp_Basic):
    def __init__(self, args):
        super(Exp_neoe, self).__init__(args)
        self.model_name = args.model
        self.data_name = args.data
    
    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTSTC,
            'TSMixer': TSMixerC,
            'DLinear': DLinearC,
            'TimesNet': TimesNetC,
            'Itransformer': Itransformer,
            'TMoE': TMoE,
            #'TMoE_v2': TMoE_v2,
            'TMoE_v3': TMoE_v3,
            'MLP' : MLP
        }
        model = model_dict[self.args.model](self.args).float()
        summary(model)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size
            data_set = Dataset_MTS(
                root_path=args.root_path,
                data_path=args.test_data_path,
                flag=flag,
                features=args.features, 
                target=args.target,
                size=[args.in_len, args.out_len],  
                data_split=args.test_data_split,
            )
            print(flag, len(data_set))
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
            data_set = Dataset_MTS(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                features=args.features, 
                target=args.target,
                size=[args.in_len, args.out_len],  
                data_split = args.data_split,
            )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        loss_f_list = []
        loss_s_list = []
        loss_cv_list = []
            
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true, cv_loss, gate_score, cls_prob = self._process_one_batch(
                    vali_data, batch_x, batch_y, if_update=False)
                loss_f = nn.MSELoss()(pred.detach().cpu(), true.detach().cpu())
                if self.args.individual == "c":
                    simMatrix = self.get_similarity_matrix_update(batch_x)
                    loss_s = self.similarity_loss_batch(self.model.cluster_prob, simMatrix)
                else: 
                    loss_s = torch.tensor(0).to(self.device)
                #if not self.args.if_moe:
                loss = loss_f + self.args.beta * loss_s
                #else:
                    #loss = loss_f + self.args.beta * loss_s + self.args.cv * cv_loss

                total_loss.append(loss.detach().item())
                loss_f_list.append(loss_f.detach().item())
                loss_s_list.append(loss_s.detach().item())
                if cv_loss is not None:
                    loss_cv_list.append(cv_loss.detach().item())
                
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
        total_loss = np.average(total_loss)
        loss_f = np.average(loss_f_list)
        loss_s = np.average(loss_s_list)
        loss_cv = np.average(loss_cv_list)
        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num
        mae, mse, rmse, mape, mspe = metrics_mean
        self.model.train()
        return mse, total_loss, loss_f, loss_s, mae
    

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        sigma=train_data.scaler.std
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()

        criterion_ts =  nn.MSELoss()
        
        
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            tl_f = []
            tl_s = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true, cv_loss, gate_score, cls_prob = self._process_one_batch(
                    train_data, batch_x, batch_y, if_update=True)
                #print(pred.shape, true.shape)
                loss_f = criterion_ts(pred, true)
                if self.args.individual == "c":
                    simMatrix = self.get_similarity_matrix_update(batch_x)
                    loss_s = self.similarity_loss_batch(self.model.cluster_prob, simMatrix)
                else: 
                    loss_s = torch.tensor(0).to(self.device)
                loss = loss_f + self.args.beta * loss_s
                #else:
                    #loss = loss_f + self.args.beta * loss_s + self.args.cv * cv_loss

                train_loss.append(loss.item())
                tl_f.append(loss_f.item())
                
                tl_s.append(loss_s.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

                #print(self.model.encoder.encoder.attn_layers[-1].Gating.weight.grad) #CHECK
                
                # if epoch % 10 == 0:
            # self.vis_linear_weight(epoch)
            
            train_loss = np.average(train_loss)
            train_loss_f = np.average(tl_f)
            train_loss_s = np.average(tl_s)
            vali_mse, vali_loss, vali_loss_f, vali_loss_s, vali_mae = self.vali(vali_data, vali_loader)
            test_mse, test_loss, test_loss_f, test_loss_s, test_mae = self.vali(test_data, test_loader)
            #print("prob", self.model.cluster_prob) #add this later

            print("Epoch: {0}, Steps: {1}, Cost time: {2:.3f} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f} Vali MSE: {6:.3f} Vali MAE: {7:.3f} Test MSE: {8:.3f} Test MAE: {9:.3f}".format(
                epoch + 1, train_steps, time.time()-epoch_time, train_loss, vali_loss, test_loss, vali_mse, vali_mae, test_mse, test_mae))
            
            wandb.log({"Train_loss":train_loss, "Train_forecast_loss":train_loss_f ,"Train_similarity_loss": train_loss_s,
                "Vali_loss": vali_loss, "Vali_forecast_loss":vali_loss_f , "Vali_similarity_loss": vali_loss_s, "Vali_mse": vali_mse,
                    "Test_loss": test_loss ,"Test_forecast_loss": test_loss_f,"Test_similarity_loss":test_loss_s, "Test_mse": test_mse,
                    "Test_mae": test_mae, "Vali_mae": vali_mae,})
            # wandb.log({"Cluster_prob": wandb.Histogram(self.model.cluster_prob.detach().cpu().numpy())})

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')
        
        return self.model

    def test(self, setting, save_pred = True, inverse = False):
        # result save
        plot_now = True
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        gate_score_lis = [] #We take record of gate score from the first layer??
        cls_prob_lis = []

        #for plotting purpose#
        n_plot = 3
        chosen_steps = sorted(np.random.choice(len(test_loader), min(n_plot, len(test_loader)), replace=False))
        #for plotting purpose#

        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                #This gate score is the gate score at final FFN layer
                pred, true, cv_loss, gate_score_lis, cls_prob= self._process_one_batch(
                    test_data, batch_x, batch_y, inverse, if_update=False)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    np_pred = pred.detach().cpu().numpy()
                    np_true = true.detach().cpu().numpy()
                    np_x = batch_x.detach().cpu().numpy()
                    preds.append(np_pred)
                    trues.append(np_true)
                    #print(gate_score)

                    #if gate_score is not None:
                        #gate_score_lis.append(gate_score.detach().cpu().numpy())
                    if cls_prob is not None:
                        cls_prob_lis.append(cls_prob.detach().cpu().numpy())

                    if i in chosen_steps:
                        if (self.args.data != "TRF") & (self.args.data != "ECL"):
                            sid = np.random.randint(0, batch_size)
                            x_seq = np_x[sid]     # [in_len, n_vars]
                            y_seq = np_true[sid]  # [out_len, n_vars]
                            y_pred = np_pred[sid] # [out_len, n_vars]
                            save_path = os.path.join(folder_path, f"sample_batch{i}_idx{sid}.png")
                            if plot_now == True:
                                plot_single_sample_all_channels(
                                    x_seq=x_seq,
                                    y_seq=y_seq,
                                    y_pred=y_pred,
                                    feature_names=test_data.all_cols,
                                    save_path=save_path,
                                    sample_idx=sid
                                )

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        mae, mse, rmse, mape, mspe = metrics_mean
        #print('mse:{}, mae:{}'.format(mse, mae))
        print(f"RESULT:: mse={mse:.4f}, mae={mae:.4f}")

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)

            title = None
            title_base = self.data_name + ' ' + self.model_name + ' '
            if len(gate_score_lis) >= 1:
            #if gate_score_lis is not None:
                np.save(folder_path + 'gate_score.npy', [g.cpu().numpy() for g in gate_score_lis])
                title = title_base + ' moe'
                #if plot_now == True:
                plot_gate_score(gate_score_lis,title_prefix=title, save_path=folder_path+title+'gate_score.png')
                #np.save(folder_path + 'gate_score.npy', [g.cpu().numpy() for g in gate_score_lis])
                #title = title_base + 'cmoe'
                #if plot_now == True:
                #    plot_gate_score(G=np.array(gate_score_lis),title_prefix=title, save_path=folder_path+title+'gate_score.png')
            if len(cls_prob_lis) >= 1:
                np.save(folder_path + 'cls_prob.npy', cls_prob_lis)
                title = title_base + 'ccm'
                if plot_now == True:
                    plot_gate_score(G=np.array(cls_prob_lis),title_prefix=title, save_path=folder_path+title+'cluster_score.png')
            
            if title is None:
                title = title_base
            

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse = False, if_update=False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        #x: [Batch, Input length, Channel]
        
        batch_size, input_length, channel = batch_x.shape

        #if not self.model.if_moe:
        if True:
            #This calculated gate score is only the gate score at final layer, for this particular batch
            outputs, gate_score_lis, cls_prob = self.model(batch_x, if_update=if_update)
            cv_loss = None

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)
        return outputs, batch_y, cv_loss, gate_score_lis, cls_prob
    
    def _similarity_loss_batch(self, prob, batch_x):
        membership = self.concrete_bern(prob)  #[n_vars, n_clusters]
        temp_1 = torch.mm(membership.t(), self.simMatrix) 
        SAS = torch.mm(temp_1, membership)
        _SS = 1 - torch.mm(membership, membership.t())
        loss = -torch.trace(SAS) + torch.trace(torch.mm(_SS, self.simMatrix)) + membership.shape[0]
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        return loss + ent_loss
    
    
    def _get_similarity_matrix(self, s_type):
        SimMatrixDict = np.load(f"temp_store/SimilarityMatrix_{self.args.data}.npy", allow_pickle=True).item()
        SimMatrix = SimMatrixDict[s_type]
        return SimMatrix
        
    
    def get_similarity_matrix(self, batch_x):
        sample = batch_x.squeeze(-1)  #[bsz, in_len]
        diff = sample.unsqueeze(1) - sample.unsqueeze(0)
        # Compute the Euclidean distance (squared)
        dist_squared = torch.sum(diff ** 2, dim=-1)  #[bsz, bsz]
        param = torch.max(dist_squared)
        euc_similarity = torch.exp(-5 * dist_squared /param )
        return euc_similarity.to(self.device) 

 
    def get_similarity_matrix_update(self,batch_data, sigma=5.0):
        """
        Compute the similarity matrix between different channels of a time series in a batch.
        The similarity is computed using the exponential function on the squared Euclidean distance
        between mean temporal differences of channels.
        
        Parameters:
            batch_data (torch.Tensor): Input data of shape (batch_len, seq_len, channel).
            sigma (float): Parameter controlling the spread of the Gaussian similarity function.
            
        Returns:
            torch.Tensor: Similarity matrix of shape (channel, channel).
        """
        batch_len, seq_len, num_channels = batch_data.shape
        similarity_matrix = torch.zeros((num_channels, num_channels), device=batch_data.device)

        # Compute point-by-point differences along the sequence length
        time_diffs = batch_data[:, 1:, :] - batch_data[:, :-1, :]  # Shape: (batch_len, seq_len-1, channel)
        
        # Compute mean of these differences over batch and sequence length
        channel_representations = time_diffs.mean(dim=(0, 1))  # Shape: (channel,)
        
        # Compute pairwise similarity
        for i in range(num_channels):
            for j in range(num_channels):
                diff = torch.norm(channel_representations[i] - channel_representations[j]) ** 2
                similarity_matrix[i, j] = torch.exp(-diff / (2 * sigma ** 2))

        return similarity_matrix.to(self.device)

     
    def similarity_loss_batch(self, prob, simMatrix):
        def concrete_bern(prob, temp = 0.07):
            random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
            prob_bern = ((prob + random_noise) / temp).sigmoid()
            return prob_bern
        membership = concrete_bern(prob)  #[n_vars, n_clusters]
        temp_1 = torch.mm(membership.t(), simMatrix) 
        SAS = torch.mm(temp_1, membership)
        _SS = 1 - torch.mm(membership, membership.t())
        loss = -torch.trace(SAS) + torch.trace(torch.mm(_SS, simMatrix)) + membership.shape[0]
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        return loss + ent_loss
        
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern
    
    
    def eval(self, setting, save_pred = True, inverse = False):
        # evaluate a saved model
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
            scale = True,
            scale_statistic = args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        gate_scores_first_batch = None

        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):
                pred, true, cv_loss, gate_score, cls_prob = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse, if_update=False)
                batch_size = pred.shape[0]
                instance_num += batch_size

                if i == 0:
                    #print("COLLECTING")
                    gate_scores_first_batch = gate_score.detach().cpu().numpy()  # [bs, n_vars, n_experts]

                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
            if gate_scores_first_batch is not None:
                np.save(folder_path + 'gate_score_batch.npy', gate_scores_first_batch)
        return mae, mse, rmse, mape, mspe

    def load_balance_loss(self, gate_output_list):
        """
        Args:
            gate_output_list (List[Tensor]): list of [n_vars, n_clusters] tensors from each layer

        Returns:
            load_balance_cv (Tensor): scalar, coefficient of variation over expert usage
        """
        # Stack all layer gate outputs → [n_layers, n_vars, n_clusters]
        stacked = torch.stack(gate_output_list, dim=0)  # shape: [n_layers, n_vars, n_clusters]

        # Sum across layers and variables → total usage per expert
        expert_usage = stacked.sum(dim=(0, 1))  # shape: [n_clusters]

        usage_mean = expert_usage.mean()
        usage_std = expert_usage.std(unbiased=False)

        return usage_std / (usage_mean + 1e-8)

    
    #alpha = 0.5 is previous setting
    def cluster_distance_loss(self, cls_emd, eps=1e-5, alpha=0.1):
        #cls_emd: [K, d]
        dists = torch.cdist(cls_emd, cls_emd, p=2)  # [K, K]
        mask = 1 - torch.eye(cls_emd.size(0), device=cls_emd.device)
        inv_dists = 1.0 / (dists + eps)
        repulsion_loss = (inv_dists * mask).mean()

        norm_reg = torch.mean(cls_emd.pow(2))  # L2 norm squared of embeddings
        return repulsion_loss + alpha * norm_reg


def plot_gate_score(gate_score_list, title_prefix="", save_path=None):
    """
    Plot mean gate scores of the first layer and argmax cluster frequencies of the last layer.

    Args:
        gate_score_list (List[np.ndarray or Tensor]): each [n_vars, n_experts]
        title_prefix (str): prefix to be added to the plot titles
        save_path (str or None): where to save the plot
    """
    # 💡 Fix here: convert any Tensor to NumPy array on CPU
    gate_score_list = [g.detach().cpu().numpy() if torch.is_tensor(g) else g for g in gate_score_list]

    assert isinstance(gate_score_list, list) and len(gate_score_list) >= 1
    assert all(g.ndim == 2 for g in gate_score_list), "Each gate score must be 2D [n_vars, n_experts]"

    first_layer = gate_score_list[0]
    last_layer = gate_score_list[-1]
    n_vars, n_experts = first_layer.shape

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(first_layer, cmap="Blues", annot=False,
                xticklabels=[f"Expert {i}" for i in range(n_experts)],
                yticklabels=[f"Channel {i}" for i in range(n_vars)],
                ax=axes[0])
    axes[0].set_title(f"{title_prefix} First Layer Gate Score")

    sns.heatmap(last_layer, cmap="Blues", annot=False,
                xticklabels=[f"Expert {i}" for i in range(n_experts)],
                yticklabels=[f"Channel {i}" for i in range(n_vars)],
                ax=axes[1])
    axes[1].set_title(f"{title_prefix} Last Layer Gate Score")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
        plt.close()
    else:
        plt.show()


def plot_single_sample_all_channels(x_seq, y_seq, y_pred, feature_names,
                                    save_path, sample_idx=0):
    """
    Plot one sample's input/output/prediction across all channels.

    Args:
        x_seq (np.ndarray): [in_len, n_vars]
        y_seq (np.ndarray): [out_len, n_vars]
        y_pred (np.ndarray): [out_len, n_vars]
        feature_names (list[str])
        save_path (str)
        sample_idx (int): used for labeling
    """
    in_len = x_seq.shape[0]
    out_len = y_seq.shape[0]
    n_vars = x_seq.shape[1]
    total_len = in_len + out_len

    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 2.5 * n_vars), sharex=True)

    if n_vars == 1:
        axes = [axes]

    for i in range(n_vars):
        ax = axes[i]
        ax.plot(range(in_len), x_seq[:, i], label="Input True", color='blue')
        ax.plot(range(in_len, total_len), y_seq[:, i], label="Output True", color='blue', linestyle='dashed')
        ax.plot(range(in_len, total_len), y_pred[:, i], label="Prediction", color='red', linestyle='dashed')

        ax.axvline(in_len, color='gray', linestyle='dotted', linewidth=1.5)

        ax.set_ylabel(feature_names[i])
        ax.grid(True)
        if i == 0:
            ax.set_title(f"Sample {sample_idx}: All Channels")
            ax.legend()

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()


def plot_tsne_by_channel(np_pred, perplexity=30, figsize=(10, 8), palette='tab10',
                         random_state=42, title="t-SNE of Samples by Channel", save_path=None):
    """
    Perform t-SNE on [bs, out_len, n_vars] prediction array and plot sample-wise t-SNE colored by channel.

    Parameters:
    - np_pred: np.ndarray of shape [bs, out_len, n_vars]
    - perplexity: t-SNE perplexity
    - figsize: matplotlib figure size
    - palette: seaborn palette for channel coloring
    - random_state: random seed for t-SNE
    - title: plot title
    - save_path: if provided, save figure to this file path (e.g., 'plot.png')
    """
    assert np_pred.ndim == 3, "Input must be of shape [bs, out_len, n_vars]"
    bs, out_len, n_vars = np_pred.shape

    # [n_vars, bs, out_len]
    x_varwise = np_pred.transpose(2, 0, 1)

    # Flatten to [n_vars * bs, out_len]
    x_reshaped = x_varwise.reshape(-1, out_len)

    # Standardize
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_reshaped)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(x_scaled)

    # Channel label for each point
    channel_labels = np.repeat(np.arange(n_vars), bs)

    # Build DataFrame
    df = pd.DataFrame({
        'TSNE-1': tsne_result[:, 0],
        'TSNE-2': tsne_result[:, 1],
        'Channel': channel_labels
    })

    # Plot
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2', hue='Channel',
                    palette=palette, s=30, alpha=0.7)
    plt.title(title)
    plt.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved t-SNE plot to {save_path}")
    else:
        plt.show()