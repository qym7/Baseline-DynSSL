from PIL import Image, ImageFilter
from skimage.transform import resize

from tqdm import tqdm
from memory_profiler import profile
import numpy as np

import torch
from torchvision import transforms


def resize_image(im, input_shape, order=1):
    if input_shape[0] != im.shape[0]:
        # im = im.astype(np.float64) * 255.
        im = resize(im, input_shape, order=order,
                    preserve_range=True, mode='constant')
        im = (im*255.).round().astype(np.uint8)
    return im


def add_texture_image(im, radius=1.5, sigma=10):
    '''
    im is numpy array
    make radius vary between 0.5 and 2.5 ?
    make sigma vary between 10 and 50 ?
    you have to normalize to [0., 1.] the result with all the data !!
    '''
    im = Image.fromarray(im)
    im = im.filter(ImageFilter.GaussianBlur(radius=radius))
    im = transforms.ToTensor()(im) * 255.
    im = torch.poisson(im)
    mu_torch = torch.ones_like(im) * 255.
    sigma_torch = torch.ones_like(im) * sigma
    im = im + torch.normal(mu_torch, sigma_torch)
    # im = (im - im.min()) / (im.max() - im.min())
    # return transforms.ToPILImage()(im)
    return im


def process_texture(data, radius=1.5, sigma=20):
    if len(data.shape) >= 3:  # data is a sequence (for 3d model)
        data = torch.stack(list(map(
                lambda im: add_texture_image(im, radius, sigma),
                data
        )))
        data = (data - data.min()) / (data.max() - data.min())
        data = list(map(
                lambda im: transforms.ToPILImage()(im),
                data
        ))
    else:  # data is an image (for 2d model)
        data = add_texture_image(data)
        data = (data - data.min()) / (data.max() - data.min())
    return data


def get_single_file(file_path,
                    data_type="synthetic",
                    label='pseudotime',
                    supervise=None):
    if data_type == "synthetic":
        npz = np.load(file_path)
        sequence = npz["segmentation_cell"].sum(axis=0)
        sequence = process_transparency(sequence)
        sequence = np.stack([resize_image(seq,
                                          (100, 100),
                                          order=1)
                             for seq in list(sequence)])
        pseudotime = npz[label]
        cell = npz["cell"]
        t0 = npz["t0"]
        pattern_info = {"t0": t0, "cell": cell, "pseudotime": pseudotime}

    elif data_type == 'livespin':
        coord = file_path.split('.')[0].split('/')[-1]
        site = file_path.split('/')[-2]
        cond = file_path.split('/')[-3]
        sequence = np.load(file_path)
        # coord = [coord] * sequence.shape[0]
        # site = [site] * sequence.shape[0]
        # cond = [cond] * sequence.shape[0]
        # pattern_info = np.stack([np.array(cond), np.array(site),
        #                          np.array(coord)])
        pattern_info = np.array([cond, site, coord])
        if supervise is not None:
            if file_path in supervise.keys():
                # if 'regression' in task_type:
                frame_event, num_cells = supervise[file_path]
                pattern_info = np.zeros(sequence.shape[0])
                for idx in range(len(frame_event)-1):
                    pattern_info[frame_event[idx]:
                                 frame_event[idx+1]] = num_cells[idx]
                pattern_info[frame_event[-1]:] = num_cells[-1]
                # elif task_type == 'supervised_split':
                #     div = supervise[file_path]
                #     pattern_info = np.zeros(sequence.shape[0])
                #     pattern_info[div] = 1
            else:
                return [None, None]

    return [sequence, pattern_info]


def process_transparency(num_cell_pixel):
    '''
    num_cell_pixel: T x H x W,
    each pixel gives the number of cells that occupies that pixel
    '''
    data = np.sqrt(num_cell_pixel)

    fg_min = 255. / data.max()

    data *= fg_min

    return data.astype(np.uint8)


def get_all_files(files_paths, data_type, label, supervise=None):
    sequences = []
    pattern_infos = []
    for file_path in tqdm(files_paths, desc='loading files',
                          total=len(files_paths)):
        data = get_single_file(file_path, data_type, label,
                               supervise=supervise)
        if data[0] is not None:
            sequences.append(data[0])
            pattern_infos.append(data[1])
    return [sequences,
            pattern_infos]


@profile
def create_datasets(all_files_paths, data_type, label, supervise=None):
    frames, seq_infos = get_all_files(all_files_paths, data_type, label,
                                      supervise)
    frames = np.concatenate(frames)
    radius = 1.5
    sigma = 10
    seq_infos = [info['pseudotime'] for info in seq_infos]
    seq_infos = np.concatenate(seq_infos).astype(int)
    frames = frames.reshape(-1, frames.shape[-2], frames.shape[-1])
    frames = torch.cat(list(map(lambda img: process_texture(
                                            img, radius, sigma), frames)))
    import matplotlib.pyplot as plt
    import random
    for i in range(20):
        img = (frames[i*20].numpy() * 255).astype(np.uint8)
        plt.imsave('data_example_{}.jpg'.format(i*20), img)
    frames = torch.cat(list(map(lambda img: torch.histc(torch.Tensor(img)
                       .reshape(-1), bins=256, min=0, max=1), frames)))
    frames = frames.reshape(-1, 256)
    return frames, seq_infos
