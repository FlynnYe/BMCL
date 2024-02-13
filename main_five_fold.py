from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from model import BMCL
from datasets.HCP import HCP_five_fold, _HCP_dataset
from datasets.OASIS import OASIS_five_fold, _OASIS_dataset
import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold
import argparse
import os


def main():

    parser = argparse.ArgumentParser(description="BMCL experiments")
    parser.add_argument('--max_epoch', type=int, default=100)  
    parser.add_argument('--batch_size', type=int, default=32)  

    # Specify the target and dataset
    parser.add_argument('--sample_size', type=int, default=164, choices=[1200, 164])
    parser.add_argument('--target', type=str, default='mmse', choices=['aggression', 'intrusiveness', 'rule_breaking', 'gender', 'AD', 'mmse'])
    parser.add_argument('--dataset', type=str, default='OASIS', choices=['OASIS', 'HCP'])

    # Specify the path to the data
    parser.add_argument('--graph_data_dir', type=str, default='./data/...')
    parser.add_argument('--label_path', type=str, default='./data/...')
    parser.add_argument('--seq_data_dir', type=str, default='./data/...')

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--merge_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--lr', type=float, default=1e-4)  
    parser.add_argument('--factor', type=float, default=0.5)  
    parser.add_argument('--patience', type=int, default=20)  

    # Whether to do the reconstruction 
    parser.add_argument('--recon', type=bool, default=False)

    args = parser.parse_args()
    print(torch.cuda.is_available())
    print(torch.cuda.device_count(), torch.cuda.get_device_name())

    pl.seed_everything(42)
    seed = torch.Generator().manual_seed(42)

    if args.dataset == 'HCP':
        full_dataset = _HCP_dataset(args=args, seq_data_dir=args.seq_data_dir, label_path=args.label_path, gdatadir=args.graph_data_dir)
    else:
        full_dataset = _OASIS_dataset(args=args, seq_data_dir=args.seq_data_dir, label_path=args.label_path, gdatadir=args.graph_data_dir)
        

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Loop over each fold
    for fold, (train_ids, val_ids) in enumerate(kfold.split(np.arange(len(full_dataset)))):
        print(f"Starting fold {fold}")

        # Setup data module for the current fold
        dm = HCP_five_fold(args=args, seed=seed) if args.dataset == 'HCP' else OASIS_five_fold(args=args, seed=seed)
        dm.prepare_data()
        dm.setup_fold(train_ids, val_ids)  

        model = BMCL(args=args)
        log_name = f"{args.target}_{args.dataset}_sample{args.sample_size}_hidden{args.hidden_dim}_layers{args.hidden_layers}_merge{args.merge_layers}_heads{args.num_heads}"
        wandb_logger = WandbLogger(log_model=False, save_dir = 'saved_models', name=log_name) 
        checkpoint_callback, defineMetricCallback = model.set_checkpoint_callback()

        trainer = Trainer(logger=wandb_logger, max_epochs=args.max_epoch, callbacks=[defineMetricCallback, checkpoint_callback], strategy='ddp_find_unused_parameters_true')

        trainer.fit(model, datamodule=dm)

        trainer.validate(ckpt_path="best", dataloaders=dm.val_dataloader())



if __name__ == '__main__':
    main()

