import os
import glob
import argparse
import numpy as np
from PIL import Image

import torch
from model import Encoder, Decoder


parser = argparse.ArgumentParser()
# Model related arguments
parser.add_argument('--encoder', default='ckpt/pre_encoder.pth',
                    help='path to load model')
parser.add_argument('--edg_decoder', default='ckpt/pre_edg_decoder.pth',
                    help='path to load model')
parser.add_argument('--cor_decoder', default='ckpt/pre_cor_decoder.pth',
                    help='path to load model')
parser.add_argument('--device', default='cuda:0',
                    help='device to run models')
# I/O related arguments
parser.add_argument('--img_glob', required=True,
                    help='Note: Remeber to quote your glob path')
parser.add_argument('--line_glob', required=True,
                    help='shold have the same number of files as img_glob'
                         'two list with same index are load as input channels'
                         'Note: Remeber to quote your glob path')
parser.add_argument('--output_dir', required=True)
# Visualization related arguments
parser.add_argument('--alpha', default=0.8,
                    help='Weight to composite output with origin rgb image')
args = parser.parse_args()
device = torch.device(args.device)


# Prepare model
encoder = Encoder().to(device)
edg_decoder = Decoder(skip_num=2, out_planes=3).to(device)
cor_decoder = Decoder(skip_num=3, out_planes=1).to(device)
encoder.load_state_dict(torch.load(args.encoder))
edg_decoder.load_state_dict(torch.load(args.edg_decoder))
cor_decoder.load_state_dict(torch.load(args.cor_decoder))


# Load path to visualization
img_paths = sorted(glob.glob(args.img_glob))
line_paths = sorted(glob.glob(args.line_glob))
assert len(img_paths) == len(line_paths), '# of input mismatch for each channels'

# Process each input
for i_path, l_path in zip(img_paths, line_paths):
    print('img  path:', i_path)
    print('line path:', l_path)

    # Load and cat input images
    i_img = np.array(Image.open(i_path), np.float32) / 255
    l_img = np.array(Image.open(l_path), np.float32) / 255
    x_img = np.concatenate([
        i_img.transpose([2, 0, 1]),
        l_img.transpose([2, 0, 1])], axis=0)

    # Feedforward and extract output images
    with torch.no_grad():
        x = torch.FloatTensor(x_img).unsqueeze(0).to(device)
        en_list = encoder(x)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])

        edg_tensor = torch.sigmoid(edg_de_list[-1].squeeze())
        cor_tensor = torch.sigmoid(cor_de_list[-1].squeeze())

        edg_img = edg_tensor.cpu().numpy().transpose([1, 2, 0])
        cor_img = cor_tensor.cpu().numpy()

    edg_img = args.alpha * edg_img + (1 - args.alpha) * i_img
    cor_img = args.alpha * cor_img[..., None] + (1 - args.alpha) * i_img

    # Dump result
    basename = os.path.splitext(os.path.basename(i_path))[0]
    edg_path = os.path.join(args.output_dir, '%sedg.png' % basename)
    cor_path = os.path.join(args.output_dir, '%scor.png' % basename)
    Image.fromarray((edg_img * 255).astype(np.uint8)).save(edg_path)
    Image.fromarray((cor_img * 255).astype(np.uint8)).save(cor_path)
