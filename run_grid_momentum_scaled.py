import os, glob
import shutil
import sys
sys.path.append("..")
import argparse
import copy
import torch as ch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from robustness import attack_steps, datasets, model_utils
from user_constants import DATA_PATH_DICT
from cox.store import Store
import try_main


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--label", help="which label to interpolate from")
	parser.add_argument("--iters", type=int,  help="how many pgd iters")
	parser.add_argument("--optims", type=str,nargs='+')
	parser.add_argument("--lrs", type=float, nargs='+')

	parser.add_argument("--record_loss", type=int, default=0, help="record loss and grad along optim path")
	parser.add_argument("--img_dir", type=str,help="path to save interpolation image")
	parser.add_argument("--loss_dir", type=str, default="",help="path to save loss/grad of interpolation image")

	parser.add_argument("--new_target",type=int,default=0)
	parser.add_argument("--target_id", type=str, default="")
	parser.add_argument("--save_to_id_prefix", type=str, help="file path to save image/rep")
	parser.add_argument("--start_from_id", type=str)

	parser.add_argument("--momentums", type=float, nargs='+')
	parser.add_argument("--momentum", type=int, default=0)
	parser.add_argument("--momentum_scaled", type=int, default=0)
	args = parser.parse_args()

	# Constants
	DATA = 'RestrictedImageNet' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
	BATCH_SIZE = 1
	NUM_WORKERS = 8
	
	# Load dataset
	dataset_function = getattr(datasets, DATA)
	dataset = dataset_function(DATA_PATH_DICT[DATA])
	_, test_loader = dataset.make_loaders(workers=NUM_WORKERS,
											  batch_size=BATCH_SIZE,
											  data_aug=False)
	data_iterator = enumerate(test_loader)
	
	# Load model
	model_kwargs = {
			'arch': 'resnet50',
			'dataset': dataset,
			'resume_path': f'/data/theory/robustopt/robust_models/imagenet_unbalanced_l2_eps30/checkpoint.pt.best',
			'parallel': False
		}
	model, _ = model_utils.make_and_restore_model(**model_kwargs)
	model.eval()
  
	for lr in args.lrs:
		for optim in args.optims:
			for momentum in args.momentums:
				name = args.save_to_id_prefix+"_"+optim+"_"+str(lr)+"lr_"+str(momentum)
				if bool(args.momentum):
					name+= "momentum"
				elif bool(args.momentum_scaled):
					name += "momentum_scaled"
				print(optim,"lr", lr, "momentum",momentum)
				try_main.main(model=model,test_loader=test_loader, label=args.label, iters=args.iters,
				save_to_id=name,img_dir=args.img_dir, 
				optim=optim, lr=lr, new_target=bool(args.new_target), target_id=args.target_id, 
				start_from_id=args.start_from_id, 
				record_loss=bool(args.record_loss), loss_dir=args.loss_dir, 
				momentum=bool(args.momentum), momentum_rate=momentum, momentum_scaled=bool(args.momentum_scaled))
   

  
