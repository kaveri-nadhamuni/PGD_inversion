import matplotlib.pyplot as plt
import numpy as np
import os
from cox.store import Store

rootdir = '/data/theory/robustopt/kaveri/inversion/inversion_loss/try/j'
prefix='j1'
optims=['adam','sgd','pgd']
lrs=[0.0001, 0.001, 0.01,0.1,1.0,10.0,100.0,1000.0]

for optim in optims:
    fig, axs = plt.subplots(4, 2)
    fig.set_size_inches(30, 15)
    fig.suptitle(optim)
    for i, lr in enumerate(lrs):
        w=i//2
        h=i - 2*w
        ax1 = axs[w, h]

        name=prefix+'_'+optim+'_'+str(lr)        
        store = Store(rootdir, name)
        input_delta_loss = store['track_optim'].df
        inp_loss = input_delta_loss['inp_loss']
        rep_loss = input_delta_loss['rep_loss']
        grad = input_delta_loss['grad']
        itr = input_delta_loss['itr']
        
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('loss')
        ax1.set_title("LR = "+str(lr)+" final rep loss = "+str(rep_loss.iloc[-1]))
        ax1.plot(itr, rep_loss, label="rep loss")
        ax1.legend()
        ax1.set_yscale('log')
        ax1.set_yticks([min(rep_loss), max(rep_loss)])
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('grad')
        ax2.plot(itr, grad, label="grad", color="green")
        ax2.legend()
        ax2.set_yscale('log')
        ax2.set_yticks([min(grad), max(grad)])
        fig.tight_layout()

    plt.savefig(os.path.join(rootdir,"all_"+optim+'.png'))

