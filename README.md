# BMCL

This repository contains the source code associated with our paper titled "Bidirectional Mapping with Contrastive Learning on Multimodal Neuroimaging Data," which has been accepted at MICCAI 2023.

## Datasets

The HCP dataset can be accessed at [here](https://www.humanconnectome.org/study/hcp-young-adult/data-releases). The OASIS dataset can be accessed at [here](https://www.oasis-brains.org/#data). The datasets should be placed under the `data/` folder.

## Installation

Ensure all the necessary packages listed in `requirements.txt` are installed.

## Running the Experiments

To conduct BMCL experiments, please execute the command below. Note that you should adjust the paths and hyperparameters according to your specific requirements:

```bash
python main.py --max_epoch 1000 --batch_size 64 --sample_size 1200 --target gender --dataset HCP --graph_data_dir './data/graphs' --label_path './data/labels.csv' --seq_data_dir './data/sequences' --hidden_dim 512 --hidden_layers 2 --merge_layers 2 --num_heads 4 --lr 1e-4 --factor 0.5 --patience 20 --recon False
```

To perform BMCL experiments with five-fold cross-validation, please execute the command below. Note that you should adjust the paths and hyperparameters according to your specific requirements:

```bash
python main_five_fold.py --max_epoch 1000 --batch_size 64 --sample_size 1200 --target gender --dataset HCP --graph_data_dir './data/graphs' --label_path './data/labels.csv' --seq_data_dir './data/sequences' --hidden_dim 512 --hidden_layers 2 --merge_layers 2 --num_heads 4 --lr 1e-4 --factor 0.5 --patience 20 --recon False
```

## Citing This Work

If you find this work useful in your research, please consider citing:

```plaintext
@inproceedings{ye2023bidirectional,
  title={Bidirectional Mapping with Contrastive Learning on Multimodal Neuroimaging Data},
  author={Ye, Kai and Tang, Haoteng and Dai, Siyuan and Guo, Lei and Liu, Johnny Yuehan and Wang, Yalin and Leow, Alex and Thompson, Paul M and Huang, Heng and Zhan, Liang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={138--148},
  year={2023},
  organization={Springer}
}

```
