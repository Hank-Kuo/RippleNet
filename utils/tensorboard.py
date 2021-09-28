import os 
import sys
from multiprocessing import Process

from torch.utils import tensorboard

class Tensorboard:
    def __init__(self, log_dir, open=True):
        self.log_dir = log_dir
        self.open = open
        if open == True:
            self.server = TensorboardServer(log_dir)
            self.server.start()
            print("Started Tensorboard Server")
            self.chrome = ChromeProcess()
            print("Started Chrome Browser")
            self.chrome.start()
        

    def create_writer(self):
        summary_writer = tensorboard.SummaryWriter(log_dir=self.log_dir)
        return summary_writer

    def delete_history(self, log_dir):
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith('events.out'):
                    file_path = os.path.join(log_dir, file)
                    os.remove(file_path)

    def finalize(self):
        if open == True:
            if self.server.is_alive():
                print('Killing Tensorboard Server')
                self.server.terminate()
                self.server.join()

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