#!/usr/bin/env python
"""
# Author: ChangXu
# Created Time : Mon 23 Apr 2021 08:26:32 PM CST
# File Name: STMAP_train.py
# Description: Modified to use Leiden clustering instead of Louvain
"""

import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt


class train():
    def __init__(self,
                processed_data,
                graph_dict,
                model,
                pre_epochs,
                epochs,
                corrupt=0.001,
                lr=5e-4,
                weight_decay=1e-4,
                domains=None,
                kl_weight=100,
                mse_weight=10,
                bce_kld_weight=0.1,
                domain_weight=1,
                use_gpu=True,
                save_path="model_checkpoints/"):
        
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.processed_data = processed_data
        self.data = torch.FloatTensor(processed_data.copy()).to(self.device)
        self.adj = graph_dict['adj_norm'].to(self.device)
        self.adj_label = graph_dict['adj_label'].to(self.device)
        self.norm = graph_dict['norm_value']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=lr, weight_decay=weight_decay)
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.num_spots = self.data.shape[0]
        self.dec_tol = 0
        self.kl_weight = kl_weight
        self.q_stride = 20
        self.mse_weight = mse_weight
        self.bce_kld_weight = bce_kld_weight
        self.domain_weight = domain_weight
        self.corrupt = corrupt
        self.save_path = save_path

        if domains is not None:
            self.domains = torch.from_numpy(domains).to(self.device)
        else:
            self.domains = domains
        
        # Setup logger
        logging.basicConfig(filename=os.path.join(save_path, 'training.log'), level=logging.INFO)
        logging.info('Training session started.')

        # Lists to track losses
        self.losses = []
        self.kl_losses = []
        self.deepst_losses = []
        
        # Create the directory for saving model checkpoints if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def process(self):
        """
        This method processes the input data through the model to get the latent
        embeddings and the soft assignment distribution 'q'.
        """
        self.model.eval()
        # Forward pass through the model to get latent variables and cluster assignments
        if self.domains is None:
            z, _, _, _, q, _, _ = self.model(self.data, self.adj)
        else:
            z, _, _, _, q, _, _, _ = self.model(self.data, self.adj)
        
        # Detach from the computation graph and move to CPU for further use
        z = z.cpu().detach().numpy()
        q = q.cpu().detach().numpy()

        return z, q

    def log_metrics(self, epoch, total_loss, kl_loss, deepst_loss):
        logging.info(f'Epoch {epoch}: Total Loss = {total_loss}, KL Loss = {kl_loss}, DeepST Loss = {deepst_loss}')
        self.losses.append(total_loss)
        self.kl_losses.append(kl_loss)
        self.deepst_losses.append(deepst_loss)

    def save_intermediate_model(self, epoch):
        model_file = os.path.join(self.save_path, f'model_epoch_{epoch}.pth')
        torch.save(self.model.state_dict(), model_file)
        logging.info(f'Model checkpoint saved at epoch {epoch}')

    def plot_losses(self):
        plt.figure()
        plt.plot(self.losses, label="Total Loss")
        plt.plot(self.kl_losses, label="KL Loss")
        plt.plot(self.deepst_losses, label="DeepST Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'training_loss_plot.png'))
        plt.show()

    def pretrain(self, grad_down=5):
        with tqdm(total=int(self.pre_epochs), 
                    desc="DeepST trains an initial model",
                        bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.pre_epochs):
                inputs_corr = masking_noise(self.data, self.corrupt)
                inputs_coor = inputs_corr.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                if self.domains is None:
                    z, mu, logvar, de_feat, _, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
                    preds = self.model.dc(z)
                else:
                    z, mu, logvar, de_feat, _, feat_x, gnn_z, domain_pred = self.model(Variable(inputs_coor), self.adj)
                    preds = self.model.model.dc(z)
                
                loss = self.model.deepst_loss(
                            decoded=de_feat, 
                            x=self.data, 
                            preds=preds, 
                            labels=self.adj_label, 
                            mu=mu, 
                            logvar=logvar, 
                            n_nodes=self.num_spots, 
                            norm=self.norm, 
                            mask=self.adj_label, 
                            mse_weight=self.mse_weight, 
                            bce_kld_weight=self.bce_kld_weight,
                            )
                if self.domains is not None:
                    loss_function = nn.CrossEntropyLoss()
                    domain_loss = loss_function(domain_pred, self.domains)
                    loss += domain_loss * self.domain_weight

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_down)
                self.optimizer.step()
                pbar.update(1)
                # Log metrics and intermediate models
                if epoch % 10 == 0:  # Log and save model every 10 epochs
                    self.log_metrics(epoch, loss.item(), kl_loss=None, deepst_loss=None)  # Modify as needed
                    self.save_intermediate_model(epoch)

    def fit(self, cluster_n=20, clusterType='Leiden', res=1.0, pretrain=True):
        if pretrain:
            self.pretrain()
            pre_z, _ = self.process()

        # Initialize y_pred_last after the first clustering step
        if clusterType == 'KMeans':
            cluster_method = KMeans(n_clusters=cluster_n, n_init=cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
        elif clusterType == 'Leiden':
            cluster_data = sc.AnnData(pre_z)
            sc.pp.neighbors(cluster_data, n_neighbors=cluster_n)
            sc.tl.leiden(cluster_data, resolution=res)
            y_pred_last = cluster_data.obs['leiden'].astype(int).to_numpy()
            n_clusters = len(np.unique(y_pred_last))
            features = pd.DataFrame(pre_z, index=np.arange(0, pre_z.shape[0]))
            Group = pd.Series(y_pred_last, index=np.arange(0, features.shape[0]), name="Group")
            Mergefeature = pd.concat([features, Group], axis=1)
            cluster_centers_ = np.asarray(Mergefeature.groupby("Group").mean())
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)

        # Regular training code
        with tqdm(total=int(self.epochs), 
                    desc="DeepST trains a final model",
                        bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.epochs):
                if epoch % self.q_stride == 0:
                    _, q = self.process()
                    q = self.model.target_distribution(torch.Tensor(q).clone().detach())
                    y_pred = q.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    if delta_label < self.dec_tol:
                        logging.info(f'Epoch {epoch}: Reached tolerance threshold. Stopping training.')
                        break

                torch.set_grad_enabled(True)
                self.model.train()
                self.optimizer.zero_grad()
                inputs_coor = self.data.to(self.device)
                z, mu, logvar, de_feat, out_q, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
                preds = self.model.dc(z)
                loss_deepst = self.model.deepst_loss(
                    decoded=de_feat, x=self.data, preds=preds, labels=self.adj_label,
                    mu=mu, logvar=logvar, n_nodes=self.num_spots, norm=self.norm,
                    mask=self.adj_label, mse_weight=self.mse_weight, bce_kld_weight=self.bce_kld_weight)
                loss_kl = F.kl_div(out_q.log(), q.to(self.device))
                total_loss = self.kl_weight * loss_kl + loss_deepst

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                # Log metrics and save model checkpoint
                self.log_metrics(epoch, total_loss.item(), loss_kl.item(), loss_deepst.item())
                if epoch % 10 == 0:
                    self.save_intermediate_model(epoch)
                
                pbar.update(1)
                
        # After training, plot losses
        self.plot_losses()


def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise
