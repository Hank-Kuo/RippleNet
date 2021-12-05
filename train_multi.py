import sys
from subprocess import check_call

PYTHON = sys.executable

def launch_training_job(model_dir, model, restore):
    if restore is not None:
        cmd = "{python} train.py --model_dir={model_dir} --model_type={model_type} --restore={restore}".format(python=PYTHON, model_dir=model_dir, model_type=model, restore=restore)
    else:
        cmd = "{python} train.py --model_dir={model_dir} --model_type={model_type}".format(python=PYTHON, model_dir=model_dir, model_type=model)
    check_call(cmd, shell=True)

if __name__ == '__main__':
    BASE_PATH = './experiments/rippleNet-movie/'
    
    folders = [   
                '''
                {'model_dir':'base_model', 'model':'base_model', 'restore': None},
                
                # kg att
                {'model_dir':'replace_model', 'model':'replace_model', 'restore': None},
                {'model_dir':'replace2_model', 'model':'replace2_model', 'restore': None},
                {'model_dir':'plus_model', 'model':'plus_model', 'restore': None},
                {'model_dir':'plus2_model', 'model':'plus2_model', 'restore': None},
                {'model_dir':'item_model', 'model':'item_model', 'restore': None},

                # hop
                {'model_dir':'hop4_head0_replace_model', 'model':'head0_replace_model', 'restore': None},
                {'model_dir':'hop4_head01_replace_model', 'model':'head01_replace_model', 'restore': None},
                {'model_dir':'hop4_head012_replace_model', 'model':'head012_replace_model', 'restore': None},
                {'model_dir':'hop4_head0123_replace_model', 'model':'head0123_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_model', 'model':'head01234_replace_model', 'restore': None},
                
                # gamma
                {'model_dir':'hop4_head01234_replace_gamma_0.1_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_gamma_0.5_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_gamma_1_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_gamma_1.5_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_gamma_2_model', 'model':'head01234_replace_model', 'restore': None},

                # cosine
                {'model_dir':'hop2_head012_replace_ouptutCosine_model', 'model':'head012_replace_ouptutCosine_model', 'restore': None},
                {'model_dir':'hop2_head012_replace_cosine_model', 'model':'head012_replace_cosine_model', 'restore': None},

                # sampler
                {'model_dir':'base_transE_v1_history_10_triplet_5_model', 'model':'base_model', 'restore': None},
                {'model_dir':'base_transE_v1_history_10_triplet_10_model', 'model':'base_model', 'restore': None},
                {'model_dir':'base_transE_v1_history_10_triplet_15_model', 'model':'base_model', 'restore': None},

                {'model_dir':'hop4_head01234_replace_transE_v1_history_10_triplet_5_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v1_history_10_triplet_10_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v1_history_10_triplet_15_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v2_history_10_triplet_5_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v2_history_10_triplet_10_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v2_history_10_triplet_15_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v3_history_10_triplet_5_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v3_history_10_triplet_10_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v3_history_10_triplet_15_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v4_history_10_triplet_5_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v4_history_10_triplet_10_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v4_history_10_triplet_15_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v5_history_10_triplet_5_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v5_history_10_triplet_10_model', 'model':'head01234_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v5_history_10_triplet_15_model', 'model':'head01234_replace_model', 'restore': None},
                
                # sample different hop
                {'model_dir':'hop0_head0_replace_transE_v3_history_10_triplet_5_model', 'model':'head0_replace_model', 'restore': None},
                {'model_dir':'hop1_head01_replace_transE_v3_history_10_triplet_5_model', 'model':'head01_replace_model', 'restore': None},
                {'model_dir':'hop2_head012_replace_transE_v3_history_10_triplet_5_model', 'model':'head012_replace_model', 'restore': None},
                {'model_dir':'hop3_head0123_replace_transE_v3_history_10_triplet_5_model', 'model':'head0123_replace_model', 'restore': None},
                {'model_dir':'hop4_head01234_replace_transE_v3_history_10_triplet_5_model', 'model':'head01234_replace_model', 'restore': None},
                '''
    ]
    
    for folder in folders:
        model_dir = BASE_PATH + folder['model_dir']
        launch_training_job(model_dir, folder['model'], folder['restore'])
    