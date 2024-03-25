import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from lpips import LPIPS
from utils import load_data_vqgan
from vqgan import VQGAN, Kernel_Conv
from torchvision import transforms

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(args.device)
        # self.kernel_conv = Kernel_Conv(args.kernel_size, 3, 3).to(args.device)

        # self.discriminator = Discriminator(args).to(device=args.device)
        # self.discriminator.apply(weights_init)
        self.vqgan.load_checkpoint('checkpoints_vqgan/finetuning/vqgan_epoch_99.pt')

        self.postprocess = transforms.Compose([
                transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225),(1/0.229, 1/0.224, 1/0.255))
        ])
        self.generate(args)


    def generate(self, args):
        # devide_ids = range(torch.cuda.device_count())
        # self.vqgan = torch.nn.DataParallel(self.vqgan,device_ids=devide_ids)
        train_dataset = load_data_vqgan(args)
        steps_one_epoch = len(train_dataset)

        with tqdm(range(len(train_dataset))) as pbar:
            for i, imgs in zip(pbar, train_dataset):
                blurry,sharp = imgs[0].to(device=args.device),imgs[1].to(device=args.device)
                kernel, _, q_loss,disc_fake = self.vqgan(blurry,sharp)

                flow_reblur = self.vqgan.kernel_estimate_gen(blurry,sharp)
                with torch.no_grad():
                    if i%2 == 0 :
                        both = torch.cat((self.postprocess(blurry[:2]),self.postprocess(sharp[:2]), self.postprocess(disc_fake[:2])))
                        vutils.save_image(both, os.path.join("results_vqgan/gen", f"finetuning_{i}.jpg"), nrow=2)

                        flow = torch.cat((self.postprocess(flow_reblur[:2]),flow_reblur[:2]))
                        vutils.save_image(flow, os.path.join("results_vqgan/gen", f"finetuning_{i}_flow.jpg"), nrow=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=361, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=4096, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=2, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--kernel-size',type=int,default=19)
    parser.add_argument('--path',type=str,default='flows_GPRO.pth')
    parser.add_argument('--finetuning',type=bool,default=False)

    args = parser.parse_args()
    args.dataset_path = r"GOPRO/train"


    train_vqgan = TrainVQGAN(args)

