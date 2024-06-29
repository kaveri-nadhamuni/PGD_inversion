import matplotlib.pyplot as plt
import numpy as np
import os
from cox.store import Store

rootdir = 'delta_loss/april12/orig_loss_single_class1_1000its_reset_odd'
lrs = [0,0.0001,0.001,0.01,0.1]

fig, axs = plt.subplots(3, 2)
for i, lr in enumerate(lrs):
    lr_store = Store(rootdir, 'lr'+str(lr))
    total_loss = lr_store['total_loss'].df
    orig_loss = lr_store['orig_loss'].df
    model_delta_loss = lr_store['model_delta_loss'].df
    input_delta_loss = lr_store['input_delta_loss'].df
    
    total_y, total_x = total_loss['loss'],total_loss['itr']
    orig_y, orig_x = orig_loss['loss'], orig_loss['itr']
    model_y, model_x = model_delta_loss['loss'], model_delta_loss['itr']
    input_y, input_x = input_delta_loss['loss'], input_delta_loss['itr']
    w,h = i%3, (i-i%3)%2
    print(w, h)  
    axs[w, h].plot(total_x, total_y, label="loss")
    axs[w,h].plot(orig_x, orig_y, label="orig loss")
    axs[w, h].plot(model_x, model_y, label="model loss")
    axs[w, h].plot(input_x, input_y, label="pgd loss")
    axs[w, h].set_title("LR = "+str(lr), {'fontsize':10})
    print(total_y[:20])
    print(model_y[:20])
    print(input_y[:20])

axs[2, 1].plot(total_x, 0*total_y, label="loss")
axs[2, 1].plot(orig_x, 0*orig_y, label="orig loss")
axs[2, 1].plot(model_x, 0*model_y, label="model loss")
axs[2, 1].plot(input_x, 0*input_y, label="pgd loss")
axs[2,1].legend(fontsize=7)
for ax in axs.flat:
    ax.set(xlabel='iters', ylabel='loss')

# Hide x labels and tick labels for top plots and y ticks for right plots.


plt.title('Inversion Loss - odd  iters update w/ reset', {'fontsize':10}, y=3.6, x=0.0)
plt.savefig('cat.png')
plt.savefig('plots/april12/orig_delta_loss_subplots_reset_odd.png')
