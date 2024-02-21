from basicsr.models import create_model
import argparse
from basicsr.utils.options import parse
from basicsr.utils import  set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist
import random
import torch
parser = argparse.ArgumentParser()
parser.add_argument(
    '-opt', type=str, default='options/test/GoPro/UFPNet-GoPro.yml', required=False, help='Path to option YAML file.')
parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

args = parser.parse_args()
opt = parse(args.opt, is_train=False)
if args.launcher == 'none':
    opt['dist'] = False
    print('Disable distributed.', flush=True)
else:
    opt['dist'] = True
    if args.launcher == 'slurm' and 'dist_params' in opt:
        init_dist(args.launcher, **opt['dist_params'])
    else:
        init_dist(args.launcher)
        print('init dist .. ', args.launcher)

opt['rank'], opt['world_size'] = get_dist_info()

# random seed
seed = opt.get('manual_seed')
if seed is None:
    seed = random.randint(1, 10000)
    opt['manual_seed'] = seed
set_random_seed(seed + opt['rank'])

if args.input_path is not None and args.output_path is not None:
    opt['img_path'] = {
        'input_img': args.input_path,
        'output_img': args.output_path
    }

model = create_model(opt)
net_g = model.net_g
flow = net_g.flow
flow = flow.to('cuda:0')
img = torch.rand((2,3,256,256)).to('cuda:0')
from basicsr.models.archs.my_module import code_extra_mean_var

kernel_extra = code_extra_mean_var(kernel_size=19).to('cuda:0')

kernel_code, _ = kernel_extra(img)
kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,                                                                                       dim=[2, 3],
                                                                       keepdim=True)

log_prob,kernel = flow.log_prob(kernel_code.reshape(kernel_code.shape[0]*kernel_code.shape[1], -1))
kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], 19, 19)
kernel_blur = kernel

kernel = kernel.permute(0, 2, 3, 1).reshape(2, 19**2, img.shape[2], img.shape[3])
log_prob = log_prob.reshape(2,img.shape[2], img.shape[3])

print(kernel.shape)
print(log_prob.shape)
# torch.save(flow.state_dict(),'flows_GPRO.pth')