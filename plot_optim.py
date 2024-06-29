import matplotlib.pyplot as plt
import numpy as np
import os
from cox.store import Store
import argparse
from textwrap import wrap

parser = argparse.ArgumentParser() 
parser.add_argument("--rootdir")  
parser.add_argument("--name")
parser.add_argument("--plot_no_grad", type=int, default=0)
parser.add_argument("--scheduler", type=int, default=0)
parser.add_argument("--acc_grad", type=int, default=0)
args = parser.parse_args()

store = Store(args.rootdir, args.name)
optim = store['track_optim'].df
inp_loss = optim['inp_loss']
rep_loss = optim['rep_loss']
grad = optim['grad']
itr = optim['itr']
		
if bool(args.scheduler):
	lr = optim['lr']

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('iterations')
	ax1.set_ylabel('loss')
	ax1.set_title("\n".join(wrap(args.name+" {0:.6g}".format(rep_loss.iloc[-1]))))
	ax1.plot(itr, rep_loss, label="real_drop")
	linthresh = (max(rep_loss)-min(rep_loss))/10
	ax1.set_yscale('symlog', linthreshy=linthresh)
	ax1.legend()

	ax2 = ax1.twinx()
	ax2.set_ylabel('LR')
	ax2.plot(itr, lr, label="LR", color="red")
	linthresh = (max(lr)-min(lr))/10
	ax2.set_yscale('log')
	ax2.legend()
	fig.tight_layout()
	plt.savefig(os.path.join(args.rootdir, args.name+'_lr_schedule.png'))
	plt.close()

if bool(args.acc_grad):
	acc_grad = optim['acc_grad']
	fig, ax1 = plt.subplots()
	ax1.set_xlabel('iterations')
	ax1.plot(itr, acc_grad, label="accumulated_grad")
	ax1.plot(itr, real_drop, label="real_drop")
	ax1.plot(itr, rep_loss, label="rep_loss")
	ax1.plot(itr, grad, label="grad")
	ax1.set_title(args.name+" rep_loss={0:.6g}".format(rep_loss.iloc[-1]))

	linthresh = (max(real_drop)-min(real_drop))/10
	ax1.set_yscale('symlog', linthreshy=linthresh)
	ax1.legend()
	plt.savefig(os.path.join(args.rootdir, args.name+'_acc_grad.png'))
	plt.close()
