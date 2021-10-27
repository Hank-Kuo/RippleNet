import sys
from subprocess import check_call

PYTHON = sys.executable

def launch_training_job(model_dir, model, restore):
    if restore is not None:
        cmd = "{python} train1.py --model_dir={model_dir} --model_type={model_type} --restore={restore}".format(python=PYTHON, model_dir=model_dir, model_type=model, restore=restore)
    else:
        cmd = "{python} train1.py --model_dir={model_dir} --model_type={model_type}".format(python=PYTHON, model_dir=model_dir, model_type=model)
    check_call(cmd, shell=True)

if __name__ == '__main__':
    BASE_PATH = './experiments/rippleNet-movie/'
    
    folders = [
          # {'model_dir':'base_model', 'model':'base_model', 'restore': None}, 
           #  {'model_dir':'replace_model', 'model':'replace_model', 'restore': None},
            #  {'model_dir':'plus_model', 'model':'plus_model', 'restore': None},
             #  {'model_dir':'plus2_model', 'model':'plus2_model', 'restore': None},
              # {'model_dir':'item_model', 'model':'item_model', 'restore': None},
              {'model_dir':'hop2_replace_model', 'model':'replace_model', 'restore': None},
              {'model_dir':'hop3_replace_model', 'model':'replace_model', 'restore': None},
              {'model_dir':'hop3_head1_replace_model', 'model':'head1_replace_model', 'restore': None},
              {'model_dir':'hop3_head2_replace_model', 'model':'head2_replace_model', 'restore': None},
              {'model_dir':'hop3_head3_replace_model', 'model':'head3_replace_model', 'restore': None},
              
'''
            {'model_dir':'head0_att_replace_gamma_0_model', 'model':'head0_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head1_att_replace_gamma_0_model', 'model':'head1_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head2_att_replace_gamma_0_model', 'model':'head2_att_replace_gamma_model', 'restore': None},
               {'model_dir':'head01_att_replace_gamma_0_model', 'model':'head01_att_replace_gamma_model', 'restore': None},
               {'model_dir':'head012_att_replace_gamma_0_model', 'model':'head012_att_replace_gamma_model', 'restore': None},
                
                # {'model_dir':'head012_att_replace_gamma_0_model', 'model':'head012_att_replace_gamma_model'},
                {'model_dir':'head012_att_replace_gamma_0.25_model', 'model':'head012_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head012_att_replace_gamma_0.5_model', 'model':'head012_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head012_att_replace_gamma_0.75_model', 'model':'head012_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head012_att_replace_gamma_1_model', 'model':'head012_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head012_att_replace_gamma_1.25_model', 'model':'head012_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head012_att_replace_gamma_1.5_model', 'model':'head012_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head012_att_replace_gamma_1.75_model', 'model':'head012_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head012_att_replace_gamma_2_model', 'model':'head012_att_replace_gamma_model', 'restore': None},

                # {'model_dir':'head0_att_replace_gamma_0_model', 'model':'head0_att_replace_gamma_model'},
                {'model_dir':'head0_att_replace_gamma_0.25_model', 'model':'head0_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head0_att_replace_gamma_0.5_model', 'model':'head0_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head0_att_replace_gamma_0.75_model', 'model':'head0_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head0_att_replace_gamma_1_model', 'model':'head0_att_replace_gamma_model', 'restore': None}, 
                {'model_dir':'head0_att_replace_gamma_1.25_model', 'model':'head0_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head0_att_replace_gamma_1.5_model', 'model':'head0_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head0_att_replace_gamma_1.75_model', 'model':'head0_att_replace_gamma_model', 'restore': None},
                {'model_dir':'head0_att_replace_gamma_2_model', 'model':'head0_att_replace_gamma_model', 'restore': None},
           '''    ]
    
    for folder in folders:
        model_dir = BASE_PATH + folder['model_dir']
        launch_training_job(model_dir, folder['model'], folder['restore'])
    