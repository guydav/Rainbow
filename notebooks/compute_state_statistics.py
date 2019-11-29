import torch
import pickle
import glob
import os
import sys
import numpy as np

device = torch.device('cuda')


def load_saved_states(seed, num=0, name='baseline-rainbow', color=False):
    state_type = color and 'color' or 'gray'
    path_glob = os.path.join('../results', f'{name}-{seed}', 'evaluation/states', f'*{num}-{state_type}.pickle')
    paths_found = glob.glob(path_glob)
    if len(paths_found) > 0:
        path = paths_found[0]

        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
                        

def resize(t):
    return torch.nn.functional.interpolate(t, (84, 84), mode='bilinear')


def extract_data(seed):
    states = [load_saved_states(seed, rep) for rep in range(10)]
    stats = []
    for state in states:
        if state is not None:
            t = resize(torch.tensor(state, device=device, dtype=torch.float).div_(255).unsqueeze(0))
            stats.append((t.squeeze().shape[0], t.mean().cpu().numpy(), t.std().cpu().numpy()))
            del(t)
            torch.cuda.empty_cache()

    return stats  
 
def combine_data(data):
    combined_length = sum([d[0] for d in data])
    combined_mean = sum(d[0] * d[1] for d in data) / combined_length
    combined_std = np.sqrt(sum([d[0] * d[2] ** 2 + d[0] * (d[1] - combined_mean) ** 2 for d in data]) / combined_length)
    return combined_length, combined_mean, combined_std


if __name__ == '__main__':
    seed = int(sys.argv[-1])
    data = extract_data(seed)
    print(combine_data(data))

