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
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq = self.configure_optimizers(args)

        self.prepare_training()
        self.postprocess = transforms.Compose([
                transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225),(1/0.229, 1/0.224, 1/0.255))
        ])
        self.train(args)

    @staticmethod
    def prepare_training():
        os.makedirs("results_vqgan", exist_ok=True)
        os.makedirs("checkpoints_vqgan", exist_ok=True)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(list(self.vqgan.flows.parameters()) +
                                  list(self.vqgan.codebook.parameters()) +
                                  list(self.vqgan.quant_conv.parameters()) +
                                  list(self.vqgan.post_quant_conv.parameters())+
                                  list(self.vqgan.kernel_conv.parameters()),
                                  lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        # opt_kernel_conv= torch.optim.Adam(self.kernel_conv.parameters(),
                                        # lr=lr, eps=1e-08,betas=(args.beta1,args.beta2)
                                    # )
        return opt_vq#, opt_kernel_conv

    def train(self, args):
        # devide_ids = range(torch.cuda.device_count())
        # self.vqgan = torch.nn.DataParallel(self.vqgan,device_ids=devide_ids)
        train_dataset = load_data_vqgan(args)
        steps_one_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    blurry,sharp = imgs[0].to(device=args.device),imgs[1].to(device=args.device)
                    kernel, _, q_loss,disc_fake = self.vqgan(blurry,sharp)

                    disc_real = blurry
                    # disc_fake = self.vqgan.conv_kernel(sharp,kernel)

                    # disc_real = self.discriminator(imgs)
                    # disc_fake = self.discriminator(decoded_images)

                    # disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch * steps_one_epoch + i,
                                                        #   threshold=args.disc_start)
                    perceptual_loss = self.perceptual_loss(disc_real, disc_fake)
                    rec_loss = torch.abs(disc_real - disc_fake)
                    nll_loss = args.perceptual_loss_factor * perceptual_loss + args.l2_loss_factor * rec_loss
                    nll_losss = nll_loss.mean()

                    # g_loss = -torch.mean(disc_fake)

                    # λ = self.vqgan.calculate_lambda(nll_losss, g_loss)
                    loss_vq = nll_losss + q_loss #+ disc_factor * λ * g_loss

                    # d_loss_real = torch.mean(F.relu(1. - disc_real))
                    # d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    # loss_gan = disc_factor * .5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    loss_vq.mean().backward(retain_graph=True)

                    # self.opt_disc.zero_grad()
                    # loss_gan.backward()

                    self.opt_vq.step()

                    # if i % 300 == 0:
                        # with torch.no_grad():
                            # both = torch.cat((blurry[:4], disc_fake[:4]))
                            # vutils.save_image(both, os.path.join("results_vqgan", f"{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(VQ_Loss=np.round(loss_vq.mean().cpu().detach().numpy().item(), 5))
                    pbar.update(0)
                with torch.no_grad():
                    both = torch.cat((self.postprocess(blurry[:4]), self.postprocess(disc_fake[:4])))
                    vutils.save_image(both, os.path.join("results_vqgan/finetuning_1024", f"{epoch}_{i}.jpg"), nrow=4)
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints_vqgan/finetuning_1024", f"vqgan_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=361, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
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

