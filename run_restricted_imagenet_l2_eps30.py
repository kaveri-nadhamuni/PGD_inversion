import os
import shutil
import sys
sys.path.append("..")
import math
import argparse
import copy
import random, string
import torch as ch
import numpy as np
from scipy import stats
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from loss_robustness import model_utils, datasets
from loss_robustness.tools.vis_tools import show_image_row, show_image_column
from user_constants import DATA_PATH_DICT
from cox.store import Store

parser = argparse.ArgumentParser()
parser.add_argument("--label", help="which label to interpolate from")
parser.add_argument("--iters", help="how many pgd iters")
parser.add_argument("--pgd_step_size", type=float, default=1.0, help="pgd learning rate")
parser.add_argument("--model_reset", type=int, default=0, help="resetting the model after updating the model weights for every grid value i.e. lr and n_group")
parser.add_argument("--second_deriv", type=int, default=0, help="using second derivative to optimize inputs")
parser.add_argument("--model_update", type=int, default=0,  help="updating model weights")
parser.add_argument("--odd_itr_update",type=int, default=0, help="updating model only in odd iterations")
parser.add_argument("--delta_loss", type=int, default=0, help="record the cumulative loss due to model updates vs pgd")
parser.add_argument("--img_path", type=str, default="",help="path to save interpolation image")
parser.add_argument("--exp_id", type=str, default="", help="subfolder to store cox delta losses")

parser.add_argument("--new_target",type=int,default=0)
parser.add_argument("--target_id", type=str, default="")
parser.add_argument("--save_to_id", type=str, help="file path to save image/rep")
parser.add_argument("--start_from_noise", type=int, default=1)
parser.add_argument("--start_from_id", type=str, default="")

parser.add_argument("--pgd", type=int, default=0)
parser.add_argument("--optim", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=0.01)
args = parser.parse_args()
print("args", args)

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
    try:
        loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
    except:
        print("rep",rep.shape,"targ",targ.shape)
    return loss, None

def img_dist(inp, targ):
    loss = ch.div(ch.norm(inp - targ, dim=1), ch.norm(targ, dim=1))
    return loss

directory = 'plots/try/'+args.target_id
loss_plot_directory = 'optim_plots/'+args.target_id
out_path = os.path.join(loss_plot_directory, 'optim_loss/')

if not os.path.exists(out_path):
    os.makedirs(out_path)

# PGD Parameters
kwargs = {
    'custom_loss': inversion_loss,
    'constraint':'2',
    'eps': 1000,
    #'step_size': 0.1 if DATA =='CIFAR' else 1,
    'step_size': args.pgd_step_size,
    'iterations': int(args.iters), 
    'targeted': True,
    'do_tqdm': True,
    #'model_update':bool(args.model_update),
    #'model_reset': bool(args.model_reset),
    #'second_deriv':bool(args.second_deriv),
    #'odd_itr_update':bool(args.odd_itr_update),
    'delta_loss':bool(args.delta_loss),
    'out_path':out_path,
    'exp_id':args.save_to_id
    }
avg_imgs, avg_reps = get_samples(test_loader, [1], label=int(args.label))
img_match, rep_match = avg_imgs[0],avg_reps[0]


name = args.target_id
if bool(args.new_target):
    shutil.rmtree(directory,  ignore_errors=True)
    if not os.path.exists(directory):
        os.makedirs(directory)
    ch.save(img_match, os.path.join(directory, name+'_target_img.pt'))
    save_image(img_match, os.path.join(directory, name+'_target_img.png'))
    ch.save(rep_match, os.path.join(directory, name+'_target_rep.pt'))
else:
    rep_match = ch.load(os.path.join(directory, name+'_target_rep.pt'))
    img_match = ch.load(os.path.join(directory, name+'_target_img.pt'))

if bool(args.start_from_noise):
    img_seed = ch.rand_like(img_match)
else:
    img_seed = ch.load(os.path.join(directory, args.start_from_id+'_try.pt'))

#print(model.model(img_seed.cuda(), with_latent=True, fake_relu=True)[1])

if not bool(args.pgd):
    store = Store(out_path, args.save_to_id, mode="w")
    store.add_table('track_optim', {'rep_loss': float,'inp_loss':float, 'grad':float,'itr': int})
    img_interpolation = img_seed.requires_grad_(True)
    iterator = range(int(args.iters))
    iterator = tqdm(iterator)
    if args.optim=="Adam":
        optim = ch.optim.Adam(params = [img_interpolation], lr=args.lr)
    else:
        optim = ch.optim.SGD(params = [img_interpolation], lr=args.lr)

    for i in iterator:
        losses,_ = inversion_loss(model.model, model.attacker.normalize(img_interpolation.cuda()), rep_match)
        loss = ch.mean(losses)
        iterator.set_description("Optim loss: {l}".format(l=loss.item()))
        
        optim.zero_grad()
        loss.backward()
        print("img grad", img_interpolation.grad.shape, ch.norm(img_interpolation.grad))
        grad = ch.norm(img_interpolation
        optim.step()
        inp_losses = img_dist(img_interpolation, img_match)
        inp_loss = ch.mean(inp_losses)
        print("inp_loss",inp_loss)
        store['track_optim'].update_row({'rep_loss':loss.item(),
                                         'inp_loss':inp_loss.item(),
                                         'grad':img_interpolation.grad.item(),
                                         'itr':i})
        store['track_optim'].flush_row()
    store.close()

else:
    (_, latent), img_interpolation = model(img_seed.cuda(), rep_match, make_adv=True, with_latent=True, **kwargs)
    loss, _ = inversion_loss(model.model, model.attacker.normalize(img_interpolation.cuda()), rep_match)
loss_str = '{0:.3g}'.format(loss.item())
print("loss",loss_str)

ch.save(img_interpolation, os.path.join(directory, args.save_to_id+"_try.pt"))
save_image(img_interpolation, os.path.join(directory, args.save_to_id+"_loss="+str(loss_str)+".png"))

