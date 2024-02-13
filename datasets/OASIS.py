import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torch
import numpy as np
import pandas as pd
import os,sys
from os.path import join

class OASIS(pl.LightningDataModule):
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
        data_set = _OASIS_dataset(args=self.args, seq_data_dir=self.seq_data_dir, label_path=self.label_path,gdatadir=self.gdatadir,
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


class OASIS_five_fold(pl.LightningDataModule):
    def __init__(self, args, seed):
        super().__init__()
        self.args = args
        self.seed = seed
        self.full_dataset = None

    def prepare_data(self):
        self.gdatadir = self.args.graph_data_dir
        self.label_path = self.args.label_path
        self.seq_data_dir = self.args.seq_data_dir
        self.full_dataset = _OASIS_dataset(args=self.args, seq_data_dir=self.seq_data_dir, label_path=self.label_path, gdatadir=self.gdatadir)


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


class _OASIS_dataset(Dataset):
    def __init__(self, args, gdatadir=None, label_path=None, seq_data_dir=None , is_train=True):
        super(_OASIS_dataset, self).__init__()
        self.args = args
        self.seq_data_dir = seq_data_dir
        self.label_path = label_path
        self.graph_dir = gdatadir

        if self.args.target == 'AD':
            self.task = 'classification'
            df = pd.read_excel(self.label_path, usecols=['MR ID', 'Primary Diagnosis'])  
            df['disease_labels'] = df['Primary Diagnosis'].apply(lambda x: 0 if x == 'Cognitively normal' else 1)
            self.labels_dict = {str(k): v for k, v in zip(list(df['MR ID']), list(df['disease_labels']))}
        elif self.args.target == 'mmse':
            self.task = 'regression'
            labelfiles = os.listdir(self.label_path)
            self.labels_dict = {}
            for i in range(len(labelfiles)):
                labeldir = os.path.join(self.label_path, labelfiles[i])
                if labelfiles[i][-4:] == '.npy':
                    subject = labelfiles[i][:-4]
                    label = np.load(labeldir)
                    self.labels_dict[subject] = label

        seqfiles = os.listdir(self.seq_data_dir)
        self.seqs_dict = {}
        for i in range(len(seqfiles)):
            seqdir = os.path.join(self.seq_data_dir, seqfiles[i])
            if seqfiles[i][-4:] == '.npy':
                subject = seqfiles[i][:-4]
                seq = np.load(seqdir).T[:, :self.args.sample_size]
                self.seqs_dict[subject] = seq

        graphfiles = os.listdir(self.graph_dir)
        self.graphs_dict = {}
        for i in range(len(graphfiles)):
            dir = os.path.join(self.graph_dir, graphfiles[i])
            if graphfiles[i][-4:] == '.npy':
                subject = graphfiles[i][:-4]
                gra = np.load(dir)
                self.graphs_dict[subject] = gra

        names = []
        adjfiles = os.listdir(self.graph_dir)
        for file in adjfiles:
            if file[-4:] == '.npy':
                names.append(file)

        self.args = args
        self.is_train = is_train
        self.names = names


    def train_status(self, status):
        if status:
            self.is_train = True
        else:
            self.is_train = False


    @property
    def labels(self):
        return [self.labels_dict[name[:-4]] for name in self.names]

    def __getitem__(self, index):
        subject = self.names[index][:-4]
        target = self.labels_dict[subject]

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