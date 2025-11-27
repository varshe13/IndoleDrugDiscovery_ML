"""
11/25/2023 12PM, BC
This file was recovered on 11/24/2023

"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import time
import os
import io
from torch.utils.tensorboard import SummaryWriter

import logging
logger_ut = logging.getLogger(__name__)



    
def viewImage(img):
    #view single tensor image

    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def viewImages(imgs):
    bigimage = make_grid(imgs)
    viewImage(bigimage)

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

def vis_OneCycleLR_scheduler(steps, epoch, lr, max_lr,dirpath):
    """
     Plot learning rate vs epoch curve
     # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling

    """
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps, epochs=epoch)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps = 100)
    lrs = []

    for i in range(epoch*steps):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    #     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.plot(lrs)
    plt.xlabel(f"epoch * steps_per_epoch = {epoch} x {steps}")
    plt.ylabel("lr")

    fname = f"lr_OneCycleLR_{steps}x{epoch}_{time.strftime('%Y%m%d%H%M')}.png"
    plt.savefig(os.path.join(dirpath,fname))
    # plt.show(block=False)
    # plt.show()


def vis_StepLR_scheduler(start_lr, end_lr, epoch,stages,dirpath):
    """
     Plot learning rate vs epoch curve
     # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling

    """
    gamma = torch.pow(10, torch.log10(torch.tensor(end_lr/start_lr))/stages)

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//stages, gamma=gamma)
    lrs = []

    for i in range(epoch):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    #     print("Factor = ",0.1 if i!=0 and i%2!=0 else 1," , Learning Rate = ",optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.plot(lrs)
    plt.title(f"Scheduler_StepLR: step={epoch//stages}, gamma={gamma:.2f}, lr={start_lr} to {end_lr}")
    plt.xlabel(f"epoch")
    plt.ylabel("lr")

    fname = f"lr_StepLR_step_{epoch//stages}_gamma_{gamma}_lr_{start_lr}to{end_lr}_T{time.strftime('%m%d')}.png"
    plt.savefig(os.path.join(dirpath,fname))


def get_obj_attributes(opt_obj):
    """  Convert an object's attributes and their values into a dict
        Used to get configuration settings in my program.

        ref: https://blog.enterprisedna.co/python-get-all-attributes

        Use __dir__ for unsorted attribute retrieval. dir() returns sorted attributes

        args: opt_obj - an object. E.g. opt of class Options
        output: a dictionary of attribute:value pairs
              the values are evaluated.
              e.g.  n_gpu = torch.cuda.device_count()
              the key value pair will be n_gpu:8  if there are 8 GPUs.

        11/22/2023, BC
    
    """
    keys = [x for x in opt_obj.__dir__() if not x.startswith('__')]
    opt_dict ={}
    tostr =''
    for k in keys:
        opt_dict[k] = getattr(opt_obj,k) # to dictionary
        tostr += f"{k} = {opt_dict[k]}\n" # to string for printing or logging
    return opt_dict, tostr



def add_scalar_multiplot(writer, plot_titles:str, index, *data):
    """  NOT Used
    plot several scaler variable in one figure.
    
    args:
        plot_titles: a string in the format of " Figure description , plot1 title, plot2 title, ...".
                    "," will be used as delimiter to separate titles
        index : x axis. e.g. epoch
        *data: arbitrary scalars.
    Usage:  add_scalar_multiplot("loss vs epoch, accuracy vs epoch, precision vs epoch", index, loss, acc, prec)
    """

    titles = plot_titles.split(",")
    titles = [t.strip() for t in titles]
    figure_description = titles[0]

    if len(data) == len(titles[1:]):
        data_dict = dict(zip(titles[1:], *data))
        writer.add_scalar(figure_description, data_dict, index)
    else:
        print(f"Error: Number of data descriptons is not equal to the number of input data")


def delete_empty_folders(root):

    deleted = set()
    
    for current_dir, subdirs, files in os.walk(root, topdown=False):

        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break
    
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            print(f" Deleting  {current_dir}")
            deleted.add(current_dir)

    return deleted




if __name__ == "__main__":

    # vis_scheduler(steps=10, epoch=10, lr=0.01, max_lr=0.05, dirpath='.')
    # rootdir = '/home/sorteratech/BC/Data/Working/Train'
    # deleted = delete_empty_folders(rootdir)
    # print(" ----------------- ")
    # print(deleted)
    pass