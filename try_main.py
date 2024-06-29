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

#Get random endpoints 
def get_samples(model, data_loader, n_groups, label=0):
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
	_, rep = model(inp, with_latent=True, fake_relu=True)
	losses = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
	return ch.mean(losses)

def img_dist(inp, targ):
	diff = inp - targ
	losses = ch.div(ch.norm(inp - targ), ch.norm(targ))
	return ch.mean(losses).item()

def create_optim(optim, params, lr):
	if optim.lower()=="pgd":
		return attack_steps.L2Step(eps=1000, orig_input=params, step_size=lr)
	elif optim.lower()=="adam":
		return ch.optim.Adam(params = [params],lr=lr)
	elif optim.lower()=="sgd":
		return ch.optim.SGD(params = [params], lr=lr)	 
	else:		 
		raise ValueError("Invalid optimizer argument")	

def step(x, g, step_size):
	l = len(x.shape) - 1
	g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
	scaled_g = g / (g_norm + 1e-10)
	return x + scaled_g * step_size

def main(model, test_loader, label, iters, save_to_id, img_dir, optim, initial_lr,
		new_target=False, target_id=None, start_from_id=None, 
		record_loss=False, loss_dir=None,
		lr_scheduler=False, lr_period=1000, lr_drop_threshold=10, lr_rise_threshold=1, lr_drop_factor=0.1, lr_restart_threshold=10e-4,
		fixed_schedule=False, relative_schedule=False,expected_schedule=False,
		momentum=False, momentum_rate=0.9, momentum_dampening=0, momentum_scaled=False,
		acc_grad=False, acc_grad_period=100):
	# generate images from dataloader
	avg_imgs, avg_reps = get_samples(model, test_loader, [1], label=int(label))
	img_match, rep_match = avg_imgs[0],avg_reps[0]

	# find starting and target images
	directory = os.path.join(img_dir, target_id)
	loss_path = os.path.join(loss_dir, target_id)

	if bool(new_target):
		print("new target")
		shutil.rmtree(directory,  ignore_errors=True)
		if not os.path.exists(directory):
			os.makedirs(directory)

		ch.save(img_match, os.path.join(directory, target_id+'_target_img.pt'))
		save_image(img_match, os.path.join(directory, target_id+'_target_img.png'))
		ch.save(rep_match, os.path.join(directory, target_id+'_target_rep.pt'))
	else:
		print("not new target")
		rep_match = ch.load(os.path.join(directory, target_id+'_target_rep.pt'))
		img_match = ch.load(os.path.join(directory, target_id+'_target_img.pt'))

	try:
		print("try to create img seed")
		img_seed = ch.load(os.path.join(directory, start_from_id+'.pt'))
		print("seed",os.path.join(directory, start_from_id+'.pt'))
	except:
		if start_from_id=="noise":
			print("Generated a new random noise")
			img_seed = ch.rand_like(img_match)
			ch.save(img_seed, os.path.join(directory, 'noise.pt'))
		else:
			raise ValueError("start_id path not found")
	
	if bool(record_loss):
		store = Store(loss_path, save_to_id, mode="w")
		columns = {'rep_loss': float,
					'inp_loss':float, 
					'grad':float,
					'itr': int}
		if lr_scheduler:
			columns.update([('lr',float), ('expected_drop',float), ('real_drop',float)])
		if acc_grad:
			columns.update([('acc_grad',float)])
		store.add_table('track_optim', columns)

	img_seed = img_seed.clone().detach().requires_grad_(True)
	img_interpolation = img_seed.clone().detach().requires_grad_(True)
	iterator = tqdm(range(int(iters)))

	rep_loss = inversion_loss(model.model, model.attacker.normalize(img_interpolation.cuda()), rep_match) 
	print("rep_loss i=0", rep_loss)

	old_drop = 0
	old_lr = initial_lr
	lr = initial_lr
	down=True
	# set the pgd/adam/sgd optimizer
	optimizer = create_optim(optim=optim, params=img_interpolation, lr=lr)	
	for i in iterator:
		if lr_scheduler:
			if i%lr_period==0:
				if i>0:
					if fixed_schedule:
						lr *= lr_scheduler_drop

					elif relative_schedule:
						print("old drop",old_drop, "real_drop_sum",real_drop_sum, "lr",lr,"old_lr",old_lr)
						if old_drop>real_drop_sum and lr!=old_lr:
							if old_lr>lr:
								down=False
							else:
								down=True
							lr, old_lr = old_lr, lr
						else:
							old_lr = lr
							if down:
								lr *= lr_scheduler_drop
							else:
								lr *= 1/lr_scheduler_drop
						old_drop = real_drop_sum
						
						
					elif expected_schedule:
						if expected_drop_sum > lr_drop_threshold*real_drop_sum:
							lr *= lr_drop_factor
						elif expected_drop_sum < lr_rise_threshold*real_drop_sum:
							lr *= 1/lr_drop_factor
						if lr < lr_restart_threshold:
							lr = initial_lr
							

				print("i",i,"lr",lr, "down",down)
				expected_drop_sum = 0
				real_drop_sum = 0
				initial_loss = rep_loss.item()
				
				if optim=="pgd":
					optimizer.lr = lr
				else:
					optimizer = create_optim(optim=optim, params=img_interpolation, lr=lr)
					  
		if optim.lower()=="pgd":			
			if momentum:
				grad_t, = ch.autograd.grad(-1*rep_loss, [img_seed])
				try:
					grad = grad*momentum_rate + grad_t*(1-momentum_dampening)
				except:
					grad = grad_t*(1-momentum_dampening)
			else:
				grad, = ch.autograd.grad(-1*rep_loss, [img_interpolation])

			if momentum_scaled:
				l = len(img_interpolation.shape) - 1
				g_norm = ch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
				scaled_grad_t = grad / (g_norm + 1e-10)
				try:
					scaled_grad = scaled_grad*momentum_rate + scaled_grad_t*(1-momentum_dampening)
				except:
					scaled_grad = scaled_grad_t*(1-momentum_dampening)
				img_interpolation = img_interpolation + scaled_grad * lr
			else:
				img_interpolation = optimizer.step(img_interpolation, grad)
 
			img_interpolation = optimizer.project(img_interpolation)
			img_interpolation = img_interpolation.clone().detach().requires_grad_(True)			   
			grad_norm = ch.norm(grad).item()	   
		else:			 
			optimizer.zero_grad()			 
			rep_loss.backward()			 
			optimizer.step()
			grad = img_interpolation.grad	
			grad_norm = ch.norm(img_interpolation.grad).item()		 
		
		inp_loss = img_dist(img_interpolation.cuda(), img_match.cuda())
		rep_loss_current = inversion_loss(model.model, model.attacker.normalize(img_interpolation.cuda()), rep_match)
		iterator.set_description('{l}'.format(l=rep_loss.item()))
		
		# tracking loss and gradients in a cox table
		row = {'rep_loss':rep_loss_current.item(),
				'inp_loss':inp_loss,
				'grad':grad_norm,
				'itr':i}
		# lr scheduler drops learning rate and plots 
		if lr_scheduler:
			expected_drop = grad_norm*lr
			real_drop = rep_loss.item() - rep_loss_current.item()
			row.update([('lr',lr), ('expected_drop',expected_drop), ('real_drop', real_drop)])
			expected_drop_sum += expected_drop
			real_drop_sum += real_drop
		rep_loss = rep_loss_current
		
		# accumulating gradients
		if acc_grad:
			if i%acc_grad_period==0:
				acc_grad_tensor = 0
			
			acc_grad_tensor += grad
			acc_grad_norm = ch.norm(acc_grad_tensor).item()
			row.update([('acc_grad', acc_grad_norm)])

		if record_loss:
			store['track_optim'].update_row(row)
			store['track_optim'].flush_row()
	if record_loss:
		store.close()
	loss_str = '{0:.6g}'.format(rep_loss.item())	
	for filename in glob.glob(os.path.join(directory, save_to_id)+"*"):		   
		os.remove(filename)		
	ch.save(img_interpolation, os.path.join(directory, save_to_id+".pt"))	 
	save_image(img_interpolation, os.path.join(directory, save_to_id+"_loss="+str(loss_str)+".png"))

if __name__ == '__main__':	  
	parser = argparse.ArgumentParser()	  
	parser.add_argument("--label", help="which label to interpolate from")	  
	parser.add_argument("--iters",type=int, help="how many pgd iters")	  
	parser.add_argument("--optim", type=str)	
	parser.add_argument("--initial_lr", type=float)   
 
	parser.add_argument("--record_loss", type=int, default=0, help="record loss and grad along optim path")    
	parser.add_argument("--img_dir", type=str,help="path to save interpolation image")	  
	parser.add_argument("--loss_dir", type=str, default="",help="path to save loss/grad of interpolation image")	
	parser.add_argument("--new_target",type=int,default=0)	  
	parser.add_argument("--target_id", type=str, default="")	
	parser.add_argument("--save_to_id", type=str, help="file path to save image/rep")	 
	parser.add_argument("--start_from_id", type=str) 

	parser.add_argument("--lr_scheduler", type=int, default=0)	  
	parser.add_argument("--lr_period", type=int)
	parser.add_argument("--lr_drop_factor", type=float)
	parser.add_argument("--lr_drop_threshold", type=float)
	parser.add_argument("--lr_rise_threshold", type=float)
	parser.add_argument("--lr_restart_threshold", type=float)

	parser.add_argument("--fixed_schedule", type=int, default=0)
	parser.add_argument("--relative_schedule", type=int, default=0)
	parser.add_argument("--expected_schedule", type=int, default=0)

	parser.add_argument("--momentum", type=int, default=0)
	parser.add_argument("--momentum_rate", type=float, default=0.9)
	parser.add_argument("--momentum_scaled", type=int, default=0)

	parser.add_argument("--acc_grad", type=int, default=0)
	parser.add_argument("--acc_grad_period", type=float, default=100)
	args = parser.parse_args()
	print(args)  
	
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
	main(model=model, test_loader=test_loader, label=args.label, iters=args.iters,			   
		save_to_id=args.save_to_id, img_dir=args.img_dir,			 
		optim=args.optim, initial_lr=args.initial_lr,			 
		new_target=bool(args.new_target), target_id=args.target_id,				
		start_from_id=args.start_from_id,			 
		record_loss=bool(args.record_loss), loss_dir=args.loss_dir,

		lr_scheduler=bool(args.lr_scheduler), lr_period=args.lr_period,
		lr_drop_factor=args.lr_drop_factor, lr_drop_threshold=args.lr_drop_threshold, lr_rise_threshold=args.lr_rise_threshold, 
		lr_restart_threshold=args.lr_restart_threshold,

		fixed_schedule=bool(args.fixed_schedule), relative_schedule=bool(args.relative_schedule), expected_schedule=bool(args.expected_schedule),

		momentum=bool(args.momentum), momentum_rate=args.momentum,
		acc_grad=bool(args.acc_grad), acc_grad_period=args.acc_grad_period
		)


