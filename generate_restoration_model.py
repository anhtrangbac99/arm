import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from restoration_model import UFPNet_code_uncertainty
from utils import load_data_vqgan, plot_images
from lr_schedule import WarmupLinearLRSchedule
from torch.utils.tensorboard import SummaryWriter
from basicsr.models.losses.losses import PSNRLoss, L1Loss
from torchvision import transforms 

class TrainRestoration:
    def __init__(self, args):
        self.model = UFPNet_code_uncertainty(args).to(device=args.device)
        self.optim = self.configure_optimizers()
        self.lr_schedule = WarmupLinearLRSchedule(
            optimizer=self.optim,
            init_lr=1e-6,
            peak_lr=args.learning_rate,
            end_lr=0.,
            warmup_epochs=10,
            epochs=args.epochs,
            current_step=args.start_from_epoch
        )
        self.postprocess = transforms.Compose([
                transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225),(1/0.229, 1/0.224, 1/0.255))
        ])
        # self.PSNRLoss = PSNRLoss()
        self.PSNRLoss = L1Loss(loss_weight=1)  
        self.L1Loss = L1Loss(loss_weight=0.01)
        if args.start_from_epoch > 1:
            self.model.load_checkpoint(args.start_from_epoch)
            print(f"Loaded Restoration Model from epoch {args.start_from_epoch}.")
        # if args.run_name:
        #     self.logger = SummaryWriter(f"./runs/{args.run_name}")
        # else:
        #     self.logger = SummaryWriter()
        self.generate(args)

    def generate(self, args):
        # devide_ids = range(torch.cuda.device_count())
        # self.model = torch.nn.DataParallel(self.model,device_ids=devide_ids)
        train_dataset = load_data_vqgan(args)
        len_train_dataset = len(train_dataset)
        step = args.start_from_epoch * len_train_dataset
        with tqdm(range(len(train_dataset))) as pbar:
            self.lr_schedule.step()
            for i, imgs in zip(pbar, train_dataset):
                imgs,sharps = imgs
                imgs,sharps = imgs.to(device=args.device),sharps.to(device=args.device)
                
                results = self.model(imgs,sharps)
                kernels, deblurs, reblurs = results[0], results[1], results[2]

                if i % 2:
                    vutils.save_image([self.postprocess(imgs[0]), self.postprocess(sharps[0]), self.postprocess(deblurs[0]), self.postprocess(reblurs[0])], os.path.join("results_res/gen", f"{i}.jpg"), nrow=4)

                step += 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=1e-4, betas=(0.9, 0.96), weight_decay=4.5e-2)
        return optimizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--latent-dim', type=int, default=361, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=64, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=8192, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--start-from-epoch', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--path',type=str,default='flows_GPRO.pth')
    parser.add_argument('--sos-token', type=int, default=1025, help='Start of Sentence token.')
    parser.add_argument('--vq-path',type=str,default='checkpoints_vqgan/vqgan_epoch_49.pt')
    parser.add_argument('--kernel-size',type=int,default=19)

    parser.add_argument('--n-layers', type=int, default=24, help='Number of layers of transformer.')
    parser.add_argument('--dim', type=int, default=768, help='Dimension of transformer.')
    parser.add_argument('--hidden-dim', type=int, default=3072, help='Dimension of transformer.')
    parser.add_argument('--num-image-tokens', type=int, default=256, help='Number of image tokens.')
    parser.add_argument('--save', type=int, default=250, help='Number of image tokens.')
    parser.add_argument('--trans-path', type=str, default='checkpoints/transformer_current.pt')
    args = parser.parse_args()
    args.run_name = "<name>"
    args.dataset_path = r"GOPRO/train"
    args.checkpoint_path = r"checkpoints_res/l1loss"
    args.n_layers = 24
    args.dim = 768
    args.hidden_dim = 3072
    args.batch_size = 8
    args.accum_grad = 25
    args.epochs = 1000

    args.start_from_epoch = 0

    args.num_codebook_vectors = 4096
    args.num_image_tokens = args.image_size**2 #* 361

    args.start_from_epoch = 600
    train_transformer = TrainRestoration(args)
    
    # x = torch.rand(1,3,9,9)#.to(device=args.device)
    
    # model(x)
    