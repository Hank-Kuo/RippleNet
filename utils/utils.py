import logging
import json 
import os
import sys
from multiprocessing import Process


import torch
from torch import nn
from torch.optim import optimizer
from typing import Tuple

_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_EPOCH = "epoch"
_BEST_SCORE = "best_score"

class Params():
    def __init__(self, json_path):
        if os.path.exists(json_path):
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        else:
            with open(json_path, 'w') as f:
                print('create json file')

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def load_checkpoint(checkpoint_dir: str, model: nn.Module, optim: optimizer.Optimizer) -> Tuple[int, int, float]:

    if not os.path.exists(checkpoint_dir):
        raise ("File doesn't exist {}".format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.tar') 
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
    optim.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    start_epoch_id = checkpoint[_EPOCH] + 1
    
    best_score = checkpoint[_BEST_SCORE]
    return start_epoch_id, best_score


def save_checkpoint(checkpoint_dir: str, model: nn.Module, optim: optimizer.Optimizer, epoch_id: int, best_score: float):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.tar') 
    
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _EPOCH: epoch_id,
        _BEST_SCORE: best_score
    }, checkpoint_path)

def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


class TensorboardSupervisor:
    def __init__(self, log_dir, history=False):
        if history:
            self.delete_history(log_dir)
        self.server = TensorboardServer(log_dir)
        self.server.start()
        print("Started Tensorboard Server")
        self.chrome = ChromeProcess()
        print("Started Chrome Browser")
        self.chrome.start()

    def delete_history(self, log_dir):
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith('events.out'):
                    file_path = os.path.join(log_dir, file)
                    os.remove(file_path)

    def finalize(self):
        if self.server.is_alive():
            print('Killing Tensorboard Server')
            self.server.terminate()
            self.server.join()
        # As a preference, we leave chrome open - but this may be amended similar to the method above

class TensorboardServer(Process):
    def __init__(self, log_dir):
        super().__init__()
        self.os_name = os.name
        self.log_dp = str(log_dir)

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" 2> NUL')
        elif self.os_name == 'posix':  # Linux
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" '
                      f'--host `hostname -I` >/dev/null 2>&1')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')
    
    
class ChromeProcess(Process):
    def __init__(self):
        super().__init__()
        self.os_name = os.name
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'start chrome  http://localhost:6006/')
        elif self.os_name == 'posix':  # Linux
            os.system(f'google-chrome http://localhost:6006/')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')