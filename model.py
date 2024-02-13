import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from loss import InfoNCE
from utils import DefineMetricCallback_cls, DefineMetricCallback_reg
from transformers import BertModel, BertConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR



class BMCL(pl.LightningModule):
    def __init__(self, args):
        super(BMCL, self).__init__()
        
        self.args = args
        self.in_dim = args.sample_size
        self.hidden_dim = args.hidden_dim
        self.hidden_layers = args.hidden_layers
        self.merge_layers = args.merge_layers
        self.num_ROIs = 82 if args.dataset == 'HCP' else 132
        self.num_heads = args.num_heads
        self.out_dim = 2 if self.args.target in ['AD','gender'] else 1
        self.recon = args.recon

        # Initialize BERT configuration
        encoder_config = BertConfig(
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.hidden_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.hidden_dim * 4,
            max_position_embeddings=self.num_ROIs,
        )
        decoder_config = BertConfig(
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.hidden_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.hidden_dim * 4,
            max_position_embeddings=self.num_ROIs,
        )
        merge_config = BertConfig(
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.merge_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.hidden_dim * 4,
            max_position_embeddings=self.num_ROIs,
        )
        self.encoder_adj = BertModel(encoder_config)
        self.encoder_seq = BertModel(encoder_config)
        self.decoder_adj = BertModel(decoder_config)
        self.decoder_seq = BertModel(decoder_config)
        self.merge = BertModel(merge_config)

        self.embedding_transform_adj = nn.Linear(self.num_ROIs, self.hidden_dim)
        self.embedding_transform_seq = nn.Linear(self.in_dim, self.hidden_dim)
        self.recon_adj = nn.Linear(self.hidden_dim, self.num_ROIs)
        self.recon_seq = nn.Linear(self.hidden_dim, self.in_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)

        self.info_nce = InfoNCE(negative_mode = 'paired')
        self.loss = nn.CrossEntropyLoss() if self.args.target in ['AD','gender'] else nn.L1Loss()
        self.metric_name = 'accuracy' if self.args.target in ['AD','gender'] else 'L1loss'
        self.metric = torchmetrics.classification.BinaryAccuracy() \
        if self.args.target in ['AD','gender'] else torchmetrics.regression.MeanAbsoluteError()

        # self.save_hyperparameters()

    def forward(self, *x):
        t_seq, t_adj = x
        seq = self.embedding_transform_seq(t_seq)
        adj = self.embedding_transform_adj(t_adj)


        _seq = self.encoder_seq(inputs_embeds=seq).last_hidden_state
        _adj = self.encoder_adj(inputs_embeds=adj).last_hidden_state

        batch_size, num_ROIs, feature_dim = _seq.shape

        negatives = generate_within_brain_negatives(seq)

        query = _seq.view(-1, feature_dim)
        positive_key = _adj.view(-1, feature_dim)
        negative_keys = negatives.view(-1, num_ROIs - 1, feature_dim)

        nce_loss = self.info_nce(query, positive_key, negative_keys)

        out = self.merge(inputs_embeds=_seq+_adj).pooler_output
        out = self.output_layer(out)

        if self.recon:
            recon_adj = self.recon_adj(self.decoder_adj(inputs_embeds=_seq).last_hidden_state)
            recon_seq = self.recon_seq(self.decoder_seq(inputs_embeds=_adj).last_hidden_state)
            recon_loss = F.l1_loss(recon_adj, t_adj, reduction='mean') + F.l1_loss(recon_seq, t_seq, reduction='mean')
        else:
            recon_loss = 0

        return out, nce_loss * 0.1, recon_loss * 0.1

    
    def set_checkpoint_callback(self):
        if self.args.target in ['AD','gender']:
            return ModelCheckpoint(monitor="val_accuracy", mode="max"), DefineMetricCallback_cls()
        else:
            return ModelCheckpoint(monitor="val_L1loss", mode="min"), DefineMetricCallback_reg()


    def training_step(self, batch, batch_idx):
        y_true, seq, adj = batch
        y_pred, nce_loss, recon_loss = self.forward(seq, adj)

        if self.args.target not in ['AD','gender']:
            y_pred = y_pred.squeeze()

        loss = self.loss(y_pred, y_true) + nce_loss + recon_loss

        if self.args.target in ['AD','gender']:
            y_pred = torch.argmax(y_pred,dim=-1)



        self.log("train_loss", loss, prog_bar=True)
        self.log(f"train_{self.metric_name}", self.metric(y_pred, y_true), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_true, seq, adj = batch
        y_pred, nce_loss, recon_loss = self.forward(seq, adj)

        if self.args.target not in ['AD','gender']:
            y_pred = y_pred.squeeze(0)

        loss = self.loss(y_pred, y_true) + nce_loss + recon_loss

        if self.args.target in ['AD','gender']:
            y_pred = torch.argmax(y_pred,dim=-1)


        self.log("val_loss", loss, prog_bar=True)
        self.log(f"val_{self.metric_name}", self.metric(y_pred, y_true), prog_bar=True)


    def configure_optimizers(self):
        # Define optimizer and optionally learning rate schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.args.factor, patience=self.args.patience, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': f"val_{self.metric_name}",  
                'interval': 'epoch',
                'frequency': 1,
            }
        }



def generate_within_brain_negatives(seq):
    batch_size, num_ROIs, feature_dim = seq.shape

    mask = ~torch.eye(num_ROIs, dtype=torch.bool, device=seq.device)
    mask = mask.unsqueeze(0).unsqueeze(-1) 
    mask = mask.expand(batch_size, -1, -1, feature_dim)  

    seq_repeated = seq.unsqueeze(1).repeat(1, num_ROIs, 1, 1)  

    negatives = seq_repeated[mask].view(batch_size, num_ROIs, num_ROIs - 1, feature_dim)

    return negatives

