import pdb
from tqdm import tqdm
from memory_profiler import profile
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader


def get_single_file(file_path, data_type="synthetic", label='pseudotime'):
    if data_type == 'syntheticbis' or data_type == "synthetic05":
        npz = np.load(file_path)
        sequence = npz["mitosis"]
        pseudotime = npz[label]
        cell = npz["cell"]
        t0 = npz["t0"]
        pattern_info = {"t0":t0, "cell":cell, "pseudotime":pseudotime}
    elif data_type == 'livespin':
        coord = file_path.split('.')[0].split('/')[-1]
        site = file_path.split('/')[-2]
        cond = file_path.split('/')[-3]
        sequence = np.load(file_path)
        #coord = [coord] * sequence.shape[0]
        #site = [site] * sequence.shape[0]
        #cond = [cond] * sequence.shape[0]
        #pattern_info = np.stack([np.array(cond), np.array(site), np.array(coord)])
        pattern_info = np.array([cond, site, coord])
        if supervise is not None:
            if file_path in supervise.keys():
                if 'regression' in task_type:
                    frame_event, num_cells = supervise[file_path]
                    pattern_info = np.zeros(sequence.shape[0])
                    for idx in range(len(frame_event)-1):
                        pattern_info[frame_event[idx]:frame_event[idx+1]] = num_cells[idx]
                    pattern_info[frame_event[-1]:] = num_cells[-1]
                elif task_type == 'supervised_split':
                    div = supervise[file_path]
                    pattern_info = np.zeros(sequence.shape[0])
                    pattern_info[div] = 1
            else:
                return [None, None]

    return [sequence, pattern_info]


def get_all_files(files_paths, data_type, label):
    sequences = []
    pattern_infos = []
    for file_path in tqdm(files_paths, desc='loading files', total=len(files_paths)):
        data = get_single_file(file_path, data_type, label)
        if data[0] is not None:
            sequences.append(data[0])
            pattern_infos.append(data[1])
    return [sequences,
            pattern_infos]

@profile
def create_datasets(all_files_paths, data_type, label):
    
    frames, seq_infos = get_all_files(all_files_paths, data_type, label)
    frames = np.concatenate(frames)
    seq_infos = [info['pseudotime'] for info in seq_infos]
    seq_infos = np.concatenate(seq_infos).astype(int)
    frames = torch.Tensor(frames)
    y_true = torch.Tensor(seq_infos)
    n_videos = y_true.shape[0]
    frames = frames.reshape(-1, frames.shape[-2], frames.shape[-1])
    frames = torch.cat(list(map(lambda img: torch.histc(torch.Tensor(img).reshape(-1), bins=256, min=0, max=255), frames)))
    frames = frames.reshape(-1, 256)
    
    return frames, seq_infos


