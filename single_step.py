import os, glob
import shutil
import sys
sys.path.append("..")
import argparse
import copy
import torch as ch
import numpy as np
import matplotlib.pyplot as plt
from robustness import attack_steps, datasets, model_utils
from user_constants import DATA_PATH_DICT
from cox.store import Store

# Custom loss for inversion
def inversion_loss(model, inp, targ):
	_, rep = model(inp, with_latent=True, fake_relu=True)
	losses = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
	return ch.mean(losses)

def step(x, g, step_size):
	l = len(x.shape) - 1
	g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
	scaled_g = g / (g_norm + 1e-10)
	return x + scaled_g * step_size

# Constants    
DATA = 'RestrictedImageNet' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']	  
BATCH_SIZE = 1	
NUM_WORKERS = 8    

#Load dataset	 
dataset_function = getattr(datasets, DATA)	  
dataset = dataset_function(DATA_PATH_DICT[DATA])	
_, test_loader = dataset.make_loaders(workers=NUM_WORKERS,batch_size=BATCH_SIZE,data_aug=False)    
data_iterator = enumerate(test_loader)	  

# Load model	
model_kwargs = {'arch': 'resnet50',		   
				'dataset': dataset,		   
				'resume_path': f'/data/theory/robustopt/robust_models/imagenet_unbalanced_l2_eps30/checkpoint.pt.best',		   
				'parallel': False}	  
model, _ = model_utils.make_and_restore_model(**model_kwargs)	 
model.eval()

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir")
parser.add_argument("--loss_dir")
parser.add_argument("--start_from_id")	  
parser.add_argument("--target_id") 
parser.add_argument("--max_step_size", type=float)
parser.add_argument("--num_intervals", type=int)
	
args = parser.parse_args()
print(args) 

start_rep = ch.load(os.path.join(args.img_dir, args.target_id, args.start_from_id+'.pt')).requires_grad_(True)
target_rep = ch.load(os.path.join(args.img_dir, args.target_id, args.target_id+'_target_rep.pt')).requires_grad_(True)

rep_loss = inversion_loss(model.model, model.attacker.normalize(start_rep.cuda()), target_rep)
print("rep loss original",rep_loss)
grad, = ch.autograd.grad(-1*rep_loss, [start_rep])

name = os.path.join(args.target_id, args.start_from_id+'_max_stepsize'+str(args.max_step_size) + '_intervals'+str(args.num_intervals))
store = Store(args.loss_dir, name, mode="w")
store.add_table('single_step', {'loss':float,
								'step_size': float})
for i in range(args.num_intervals):
	step_size = (i/args.num_intervals)*args.max_step_size
	updated_rep = start_rep + step_size*grad
	updated_loss = inversion_loss(model.model, model.attacker.normalize(updated_rep.cuda()), target_rep)
	print('i',i,'step_size',step_size,'updated_loss',updated_loss.item())
	row = {'loss': updated_loss.item(), 'step_size': step_size}
	store['single_step'].update_row(row)
	store['single_step'].flush_row()

store.close()
