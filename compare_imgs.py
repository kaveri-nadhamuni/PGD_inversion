import os
import shutil
import sys
sys.path.append("..")
import argparse
import torch as ch
import numpy as np
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--seed_path", help="path for seed image that was tranformed")
parser.add_argument("--out_path", help="path for outputted image that was the result of a tranformation")
parser.add_argument("--diff_path", help="save difference as an image")
args = parser.parse_args()

def compare(inp, out, diff_path):
    diff = out.cuda()-inp.cuda()
    print(diff.shape)
    print("mean diff:",diff.mean())
    print(diff.argmax())
    save_image(diff, diff_path)

inp = ch.load(args.seed_path)
out = ch.load(args.out_path)
compare(inp, out, args.diff_path)
