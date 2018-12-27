import torch

from scipy.misc import imread, imsave, imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

import torch.nn.functional as F
from models import DepthNet
from util import tensor2array

parser = argparse.ArgumentParser(description='Export model using torch.jit.trace.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--img-height", default=512, type=int, help="Image height")
parser.add_argument("--img-width", default=512, type=int, help="Image width")
# parser.add_argument("--cuda", default=False, type=bool, help="use GPU")

parser.add_argument("--pretrained", required=True, type=str, help="pretrained DepthNet path")

@torch.no_grad()
def main():
    args = parser.parse_args()
    # if args.cuda:
    #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #     mode = "gpu" if torch.cuda.is_available() else "cpu"
    # else:
    #     device = torch.device("cpu")
    #     mode = "cpu"

    device = torch.device("cpu")
    mode = "cpu"

    weights = torch.load(args.pretrained, map_location='cpu')
    depth_net = DepthNet(batch_norm=weights['bn'],
                         depth_activation=weights['activation_function'],
                         clamp=weights['clamp']).to(device)
    depth_net.load_state_dict(weights['state_dict'])
    depth_net.eval()

    h = args.img_height 
    w = args.img_width
    # input-tensor-size = 1 x 6 x h x w
    traced_net = torch.jit.trace(depth_net, torch.rand(1, 6, h, w).to(device))
    traced_net.save("DepthNet_h{}_w{}_{}.pt".format(h,w,mode))
    print("DepthNet_h{}_w{}_{}.pt is exported".format(h,w,mode))

if __name__ == '__main__':
    main()
