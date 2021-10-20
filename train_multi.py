import sys
from subprocess import check_call

PYTHON = sys.executable

def launch_training_job(model_dir, model):
    cmd = "{python} train.py --model_dir={model_dir} --model_type={model_type}".format(python=PYTHON, model_dir=model_dir, model_type=model)
    check_call(cmd, shell=True)

if __name__ == '__main__':
    BASE_PATH = './experiments/rippleNet-movie/'
    
    folders = [{'model_dir':'basic_model', 'model':'basic_model'}, {'model_dir':'head0_model', 'model':'head0_model'}, 
                {'model_dir':'head1_model', 'model':'head1_model'}, {'model_dir':'head2_model', 'model':'head2_model'},
                {'model_dir':'plus_model', 'model':'plus_model'}]
    
    for folder in folders:
        model_dir = BASE_PATH + folder['model_dir']
        launch_training_job(model_dir, folder['model'])
    