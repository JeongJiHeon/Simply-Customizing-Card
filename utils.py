import torch
import random
import os
import matplotlib.pyplot as plt
import numpy as np

def make_fix_idx(num, total):
    idx = []
    while True:
        rand = random.randint(0, total)
        if not rand in idx:
            idx.append(rand)
            
        if len(idx) == num:
            break
    return idx


def make_fix_img(idx, dataset):
    return torch.cat([dataset[idx_].unsqueeze(0) for idx_ in idx])


def saveimage(image, num, model='A', ID = 1, figsize = (16,16), x=4, y=4, label = True, name = True):
    path = os.getcwd()
    path += '/{}/output{}/{:03}k_{}_.jpg'.format(ID, model, num, model)
    image = image.cpu().detach().numpy() +1
    image /= 2
    fig, ax = plt.subplots(x, y, figsize=figsize)
    
    image = np.transpose(image, (0, 2, 3, 1))
    
    for i, img in enumerate(image):
        ax[int(i/4), i%4].imshow(img)
        ax[int(i/4), i%4].get_xaxis().set_visible(False)
        ax[int(i/4), i%4].get_yaxis().set_visible(False)

    if name:
        del(name)
        name = 'iter {:03}_{}'.format(num, model)
    if label:
        fig.text(0.5, 0.04, name, ha='center', fontsize = 15)
    plt.savefig(path)
    del(fig)
    del(ax)
    plt.close()