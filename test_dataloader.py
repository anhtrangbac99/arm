from utils import load_data
import argparse


parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument('--run-name', type=str, default=None)
parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension n_z.')
parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
parser.add_argument('--num-codebook-vectors', type=int, default=8192, help='Number of codebook vectors.')
parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
parser.add_argument('--dataset-path', type=str, default='./GOPRO_S', help='Path to data.')
parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')
parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--start-from-epoch', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--ckpt-interval', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

parser.add_argument('--sos-token', type=int, default=1025, help='Start of Sentence token.')

parser.add_argument('--n-layers', type=int, default=24, help='Number of layers of transformer.')
parser.add_argument('--dim', type=int, default=768, help='Dimension of transformer.')
parser.add_argument('--hidden-dim', type=int, default=3072, help='Dimension of transformer.')
parser.add_argument('--num-image-tokens', type=int, default=256, help='Number of image tokens.')

args = parser.parse_args()
args.run_name = "<name>"
args.dataset_path = r".\\GOPRO_S\\blur"
args.checkpoint_path = r".\checkpoints"
args.n_layers = 24
args.dim = 768
args.hidden_dim = 3072
args.batch_size = 4
args.accum_grad = 25
args.epochs = 1000

args.start_from_epoch = 0

args.num_codebook_vectors = 1024
args.num_image_tokens = 256


dataset = load_data(args)

img = next(iter(dataset))

print(img.shape)