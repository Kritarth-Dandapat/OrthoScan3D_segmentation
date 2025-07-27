import argparse
import random
import numpy as np
import os

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data.dataset import Dataset

from dataset.mesh_dataset import Teeth3DSDataset
from dataset.preprocessing import *
from models.dilated_tooth_seg_network import LitDilatedToothSegmentationNetwork

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(SEED)

torch.set_float32_matmul_precision('medium')

random.seed(SEED)

seed_everything(SEED, workers=True)


def get_model():
    return LitDilatedToothSegmentationNetwork()


def get_dataset(train_test_split=1) -> tuple[Dataset, Dataset]:

    train = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed',
                                       verbose=True,
                                       pre_transform=PreTransform(classes=17),
                                       post_transform=None, in_memory=False,
                                       force_process=False, is_train=True, train_test_split=train_test_split)
    test = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed',
                                      verbose=True,
                                      pre_transform=PreTransform(classes=17),
                                      post_transform=None, in_memory=False,
                                      force_process=False, is_train=False, train_test_split=train_test_split)

    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Training with Wandb')
    parser.add_argument('--epochs', type=int,
                        help='How many epochs to train', default=100)
    parser.add_argument('--tb_save_dir', type=str,
                        help='Tensorboard save directory', default='tensorboard_logs')
    parser.add_argument('--experiment_name', type=str,
                        help='Experiment Name', default='teeth_segmentation')
    parser.add_argument('--experiment_version', type=str,
                        help='Experiment Version', default='v1')
    parser.add_argument('--train_batch_size', type=int,
                        help='Train batch size', default=2)
    parser.add_argument('--devices', nargs='+', help='Devices to use', required=True, default=[0])
    parser.add_argument('--n_bit_precision', type=int,
                        help='N-Bit precision', default=16)
    parser.add_argument('--train_test_split', type=int,
                        help='Train test split option. Either 1 or 2', default=1)
    parser.add_argument('--ckpt', type=str,
                        required=False,
                        help='Checkpoint path to resume training', default=None)
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging', default=True)
    parser.add_argument('--wandb_project', type=str,
                        help='Wandb project name', default='teeth-segmentation-3d')
    parser.add_argument('--wandb_entity', type=str,
                        help='Wandb entity/username', default=None)

    args = parser.parse_args()

    print(f'Run Experiment using args: {args}')

    # Initialize wandb if requested
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(args.tb_save_dir, name=args.experiment_name, version=args.experiment_version)
    loggers.append(tb_logger)
    
    # Wandb logger
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.experiment_name}_{args.experiment_version}",
            log_model=True,
            save_dir=args.tb_save_dir
        )
        loggers.append(wandb_logger)
        
        # Log hyperparameters
        wandb_logger.log_hyperparams({
            'epochs': args.epochs,
            'batch_size': args.train_batch_size,
            'precision': args.n_bit_precision,
            'train_test_split': args.train_test_split,
            'devices': args.devices,
            'seed': SEED
        })

    train_dataset, test_dataset = get_dataset(args.train_test_split)

    model = get_model()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                                   shuffle=True, drop_last=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    log_dir = tb_logger.log_dir

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir, 
        save_top_k=3, 
        monitor="val_acc", 
        mode='max',
        filename='teeth-seg-{epoch:02d}-{val_acc:.3f}'
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        accelerator='cuda', 
        devices=[int(d) for d in args.devices],
        enable_progress_bar=True, 
        logger=loggers, 
        precision=args.n_bit_precision,
        callbacks=[checkpoint_callback], 
        deterministic=False,
        log_every_n_steps=10
    )
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.ckpt) 