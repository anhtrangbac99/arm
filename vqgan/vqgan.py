import torch
import torch.nn as nn
# from encoder import Encoder
# from decoder import Decoder
from codebook import Codebook
from basicsr.models.archs.my_module import code_extra_mean_var
from torch.nn import functional as F

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        # self.encoder = Encoder(args).to(device=args.device)
        # self.decoder = Decoder(args).to(device=args.device)

        self.kernel_extra = code_extra_mean_var(kernel_size=19)
        self.kernel_size = 19
        self.flows = self.load_flows(args)

        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

    @staticmethod
    def load_flows(args):
        from basicsr.models.archs.Flow_arch import KernelPrior
        flow = KernelPrior(n_blocks=5, input_size=19 ** 2, hidden_size=25, n_hidden=1, kernel_size=19)
        flow.load_state_dict(torch.load(args.path))
        for param in flow.parameters():
            param.requires_grad = False 
        flow = flow.eval()
        return flow
    
    @torch.no_grad()
    def flow_to_log_prob(self,x):
        log_prob,_ = self.flows.log_prob(x)
        k,_ = self.flows.inverse(x)
        return k,-log_prob
    
    def kernel_estimate(self,x):
            B,C,H,W = x.shape
            kernel_code, _ = self.kernel_extra(x)
            kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
                                                                                                        dim=[2, 3],
                                                                                                        keepdim=True)
            # # code uncertainty
            # sigma = kernel_var
            # kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(kernel_code) * sigma

            kernel,nev_log_prb = self.flow_to_log_prob(kernel_code.reshape(kernel_code.shape[0]*kernel_code.shape[1], -1))
            kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
            # kernel_blur = kernel

            kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size*self.kernel_size, x.shape[2], x.shape[3]) #Bx(19*19)xHxW
            
            return kernel
    
    def forward(self, imgs):

        kernel = self.kernel_estimate(imgs)
        # encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(kernel)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(quantized_codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, x):
        encoded_images = self.encoder(x)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        quantized_codebook_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_codebook_mapping)
        return decoded_images

    def calculate_lambda(self, nll_loss, g_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        print("Loaded Checkpoint for VQGAN....")


class kernel_cov(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_cov, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_kernel = nn.Sequential(
                        nn.Conv2d(kernel_size*kernel_size, out_ch, kernel_size=1, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=1, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)


        x = F.conv2d(x,kernel)
    

        return output