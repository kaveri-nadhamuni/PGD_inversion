import matplotlib.pyplot as plt
import numpy as np
import os
from cox.store import Store
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--rootdir")  
parser.add_argument("--name")
args = parser.parse_args()

store = Store(args.rootdir, args.name)
optim = store['single_step'].df
loss = optim['loss']
step_size = optim['step_size']
		
plt.xlabel('step_size')
plt.ylabel('loss')
plt.title(args.name+" rep_loss={0:.6g}".format(loss.iloc[-1]))
plt.plot(step_size, loss, label="rep_loss")
print(step_size, loss)
plt.legend()

plt.savefig(os.path.join(args.rootdir, args.name+'.png'))
plt.close()