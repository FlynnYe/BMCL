import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torch
import numpy as np
import pandas as pd
import os,sys
from os.path import join

class HCP(pl.LightningDataModule):
    def __init__(self, args, seed):
        super().__init__()
        self.args = args
        self.seed = seed

    def prepare_data(self):
        self.gdatadir = self.args.graph_data_dir
        self.label_path = self.args.label_path
        self.seq_data_dir = self.args.seq_data_dir

    def setup(self, stage=None):
        # Load and split the dataset into train, val, and test sets
        data_set = _HCP_dataset(args=self.args, seq_data_dir=self.seq_data_dir, label_path=self.label_path,gdatadir=self.gdatadir,
                               is_train=True)
        train, val = _dataset_train_test_split(
            data_set, train_size=0.8, generator=self.seed
        )
        train.train_status = True
        val.train_status = False
        self.train_dataset = train
        self.val_dataset = val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)


class HCP_five_fold(pl.LightningDataModule):
    def __init__(self, args, seed):
        super().__init__()
        self.args = args
        self.seed = seed
        self.full_dataset = None

    def prepare_data(self):
        self.gdatadir = self.args.graph_data_dir
        self.label_path = self.args.label_path
        self.seq_data_dir = self.args.seq_data_dir
        self.full_dataset = _HCP_dataset(args=self.args, seq_data_dir=self.seq_data_dir, label_path=self.label_path, gdatadir=self.gdatadir)


    def setup(self, stage=None):
        pass
    
    def setup_fold(self, train_ids, val_ids):
        # Use the train_ids and val_ids to create dataset subsets for the current fold
        self.train_dataset = Subset(self.full_dataset, train_ids)
        self.val_dataset = Subset(self.full_dataset, val_ids)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)


class _HCP_dataset(Dataset):
    def __init__(self, args, gdatadir=None, label_path=None, seq_data_dir=None , is_train=True):
        super(_HCP_dataset, self).__init__()
        self.args = args
        self.seq_data_dir = seq_data_dir
        self.label_path = label_path
        self.gdatadir = gdatadir

        self.pdlabels = pd.read_csv(self.label_path)
        pdsubjedt = self.pdlabels['Subject']

        if self.args.target == 'gender':
            pgender = self.pdlabels['Gender']
            self.task = 'classification'

            if self.args.target == 'gender':
                ptarget = pgender
                ptarget = ptarget.apply(lambda x: 0 if x == 'F' else 1)
                
            else:
                print('error!!')
                sys.exit(1)
        else:
            self.task = 'regression'

            paggre = self.pdlabels['ASR_Aggr_Pct']
            prule = self.pdlabels['ASR_Rule_Pct']
            pintr = self.pdlabels['ASR_Intr_Pct']

            if self.args.target == 'aggression':
                ptarget = paggre
            elif self.args.target == 'intrusiveness':
                ptarget = pintr
            elif self.args.target == 'rule_breaking':
                ptarget = prule
            else:
                print('error!!')
                sys.exit(1)


        _target_dict = {str(k): v for k, v in zip(list(pdsubjedt), list(ptarget))}
        target_dict = {}
        for k, v in _target_dict.items():
            if not np.isnan(v):
                target_dict[k] = v

        sequence_dir = self.seq_data_dir + '/QCed_sequence'
        seqfiles = os.listdir(sequence_dir)
        self.seqs_dict = {}
        for i in range(len(seqfiles)):
            seqdir = os.path.join(sequence_dir, seqfiles[i])
            if seqfiles[i][-4:] == '.npy':
                subject = seqfiles[i][:-4]
                seq = np.load(seqdir).T[:, :self.args.sample_size]
                self.seqs_dict[subject] = seq
        self.filtered_target_dict = {}
        for k, v in self.seqs_dict.items():
            if k not in list(target_dict.keys()):
                continue

            elif v.shape[1] == self.args.sample_size and v.shape[0] == 82:
                self.filtered_target_dict[k] = target_dict[k]
        # self.subject = list(self.filtered_target_dict.keys())

        # Load Graph Data
        graph_dir = join(self.gdatadir)
        names = []
        adjfiles = os.listdir(graph_dir)
        for file in adjfiles:
            if file[-4:] == '.npy':
                names.append(file)

        self.graph_dir = graph_dir
        self.is_train = is_train

        # names = self.names
        self.names = []
        for name in names:
            if name[:-4] not in list(self.filtered_target_dict.keys()):
                continue
            else:
                self.names.append(name)

    def train_status(self, status):
        if status:
            self.is_train = True
        else:
            self.is_train = False


    @property
    def labels(self):
        return [self.filtered_target_dict[name[:-4]] for name in self.names]

    def __getitem__(self, index):
        subject = self.names[index][:-4]
        target = self.filtered_target_dict[subject]

        seq = self.seqs_dict[subject]
        seq = torch.from_numpy(seq).float()

        adj = np.load(join(self.graph_dir, self.names[index]))

        adj = torch.from_numpy(adj).float()
        if self.task == 'classification':
            target = torch.tensor(target).long()
        else:
            target = torch.tensor(target).float()

        return target, seq, adj
    
    def __len__(self):
        return len(self.names)




def _dataset_train_test_split(
    dataset, train_size: float, generator: torch.Generator,
):

    num_items = len(dataset)  
    num_train = round(num_items * train_size)
    permutation = torch.randperm(num_items, generator=generator)
    return (
        Subset(dataset, permutation[:num_train].tolist()),
        Subset(dataset, permutation[num_train:].tolist()),
    )