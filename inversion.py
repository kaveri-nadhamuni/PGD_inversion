import os
import sys
sys.path.append("..")
import math
import argparse
import copy
import torch as ch
import numpy as np
from scipy import stats
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from krobustness import model_utils, datasets
from krobustness.tools.vis_tools import show_image_row, show_image_column
from user_constants import DATA_PATH_DICT

parser = argparse.ArgumentParser()
parser.add_argument("--label", help="which label to interpolate from")
parser.add_argument("--iters", help="how many pgd iters")
parser.add_argument("--reset", choices=["grid","row","column","never"],help="when to reset model")
parser.add_argument("--row_by_row", type=int, default=0, help="if the model is never reset, should it go through the grid row-by-row or column-by-column")
parser.add_argument("--img_avg", type=int, default=0, help="if the x axis should be n=1,2,4,...,|n| or random images")
parser.add_argument("--num_imgs", type=int, default=5, help="number of random images along x axis")
parser.add_argument("--lrs",default=[0,0.0001,0.001,0.01,0.1],type=float,nargs='+',help="lrs to try along vertical axis") 
parser.add_argument("--model_reset", type=int, default=0, help="resetting the model after updating the model weights for every grid value i.e. lr and n_group")
parser.add_argument("--second_deriv", type=int, default=0, help="using second derivatives to optimize inputs")
parser.add_argument("--model_update", type=int, default=0,  help="updating model weights")
parser.add_argument("--odd_itr_update",type=int, default=0, help="updating model only in odd iterations")
parser.add_argument("--no_pgd", type=int, default=0, help="no updating image w pgd")
parser.add_argument("--img_path", help="file path to save interpolation image")
args = parser.parse_args()


# Constants
DATA = 'RestrictedImageNet' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
BATCH_SIZE = 1
NUM_WORKERS = 8
NOISE_SCALE = 20
NUM_INTERPOLATES = 10


DATA_SHAPE = 32 if DATA == 'CIFAR' else 224 # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048 # Size of representation vector (fixed for model)

# Load dataset
dataset_function = getattr(datasets, DATA)
dataset = dataset_function(DATA_PATH_DICT[DATA])
#dataset = dataset_function('/scratch/engstrom_scratch/imagenet')
_, test_loader = dataset.make_loaders(workers=NUM_WORKERS, 
                                      batch_size=BATCH_SIZE, 
                                      data_aug=False)
data_iterator = enumerate(test_loader)

counts = {2: 150, 3: 250, 4: 1050, 7: 200, 0: 5900, 1: 250, 8: 1000, 5: 900, 6: 450}

# Load model
model_kwargs = {
    'arch': 'resnet50',
    'dataset': dataset,
    'resume_path': f'/data/theory/robustopt/robust_models/imagenet_unbalanced_l2_eps30/checkpoint.pt.best',
    'parallel': False
}
model, _ = model_utils.make_and_restore_model(**model_kwargs)
model.eval()

#Get random endpoints 
def get_samples(data_loader, n_groups, label=0):
    avg_imgs = []
    avg_reps = []
    it = enumerate(data_loader)
    for n in n_groups:
        print("get samples n",n)
        _, (img, _) = next(it)
        with ch.no_grad():
            (_, rep), _ = model(img.cuda(), with_latent=True)
        sum_img = ch.zeros_like(img)
        sum_rep = ch.zeros_like(rep.detach())
        count=0
        while count<n:
            try:
                _, (im_s, im_label) = next(it)
            except:
                it = enumerate(data_loader)
                _, (im_s, im_label) = next(it)
            
            if im_label!=label:
                continue
            count+=1
            with ch.no_grad():
                (_, rep_s), _ = model(im_s.cuda(), with_latent=True)
            sum_img += im_s
            sum_rep += rep_s.detach() 
        avg_imgs.append(sum_img/n)
        avg_reps.append(sum_rep/n)
        
    return avg_imgs, avg_reps

# Custom loss for inversion
def inversion_loss(model, inp, targ):
    (_, rep) = model(inp, with_latent=True, fake_relu=True)
    loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
    return loss, None

# PGD Parameters
kwargs = {
    'custom_loss': inversion_loss,
    'constraint':'2',
    'eps': 1000,
    'step_size': 0.1 if DATA =='CIFAR' else 1,
    'iterations': int(args.iters), 
    'targeted': True,
    'do_tqdm': True,
    'model_update':bool(args.model_update),
    'model_reset': bool(args.model_reset),
    'second_deriv':bool(args.second_deriv),
    'odd_itr_update':bool(args.odd_itr_update),
    'no_pgd':bool(args.no_pgd)
}

# get n_groups and lrs based on parameters
lrs = args.lrs
if args.img_avg==1:
    n_class = counts[int(args.label)]
    n_groups = [1]*n_class
    n_groups = [2**i if 2**i<=n_class else n_class for i in range(math.ceil(math.log(n_class, 2)+1))]
else:
    n_groups = [1]*args.num_imgs

# for the image and loss grid
inversion_grid = [[-1 for _ in range(len(n_groups))] for _ in range(len(lrs)+1)]
each_title = [['' for _ in range(len(n_groups))] for _ in range(len(lrs)+1)]
orig_model = copy.deepcopy(model)

# iteration order
if args.reset=="row" or bool(args.row_by_row):
    outer = lrs
    inner = n_groups
    row_flag = True
else:
    outer = n_groups
    inner = lrs
    row_flag = False

# generate list of average images and representations for all n 
avg_imgs, avg_reps = get_samples(test_loader, n_groups, label=int(args.label))
inversion_grid[0] = avg_imgs.copy()
img_seed = ch.rand_like(avg_imgs[0])

for i in range(len(outer)):
    if args.reset=="row" or args.reset=="column":
        model = copy.deepcopy(orig_model)

    for j in range(len(inner)):
        if args.reset=="grid":
            model = copy.deepcopy(orig_model)

        if row_flag:
            _, rep_match = avg_imgs[j], avg_reps[j]
            n = n_groups[j]
            lr = lrs[i]
            kwargs['model_lr'] = lrs[i]
            grid_outer, grid_inner = i+1, j 
        else:
            _, rep_match = avg_imgs[i], avg_reps[i]
            n = n_groups[i]
            lr = lrs[j]
            kwargs['model_lr'] = inner[j]
            grid_outer, grid_inner = j+1, i

        # go from avg image to avg representation
        (_, latent), img_interpolation = model(img_seed.cuda(), rep_match, make_adv=True, with_latent=True,**kwargs)

        # calculate loss of updated model
        updated_model_loss, _ = model.attacker.calc_loss(img_interpolation.cuda(), rep_match, custom_loss=inversion_loss, should_normalize=True)

        # calculate loss of original model
        orig_model_loss,_ = orig_model.attacker.calc_loss(img_interpolation.cuda(), rep_match, custom_loss=inversion_loss, should_normalize=True)
        loss_str, orig_loss_str = '{0:.3g}'.format(updated_model_loss.item()), '{0:.3g}'.format(orig_model_loss.item())
        print("grid_outer",grid_outer,"grid_inner",grid_inner,"lr",lr,"n",n,"updated model loss", loss_str, "orig loss", orig_loss_str) 
        inversion_grid[grid_outer][grid_inner] = img_interpolation
        each_title[grid_outer][grid_inner] = 'loss=' + loss_str + ' orig_loss=' + orig_loss_str

# post processing for making the image grid
row_title = ['lr='+str(i) for i in lrs]
row_title.insert(0, 'img_seed')

inversion_tensors = []
device = ch.device("cuda")
for row in inversion_grid:
    cat_row = ch.cat([r.to(device) for r in row])
    inversion_tensors.append(cat_row.detach().cpu())
  
show_image_row(inversion_tensors,
               row_title,
               tlist=each_title,
               fontsize=9,
               filename=args.img_path)
