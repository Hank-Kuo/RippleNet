"""Peform hyperparemeters search"""
import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./experiments/base_model', help='Directory containing params.json')
parser.add_argument('--params_dir', default='./experiments/search_params', help="Directory containing the dataset")


def launch_training_job(parent_dir, job_name, params):
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir}".format(python=PYTHON, model_dir=model_dir)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    args = parser.parse_args()
    utils.check_dir(args.params_dir)
    params_path = os.path.join(args.model_dir, 'params.json')
    params = utils.Params(params_path)
    
    # Perform hypersearch over one parameter
    learning_rates = [0.1, 0.075, 0.05]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.params_dir, job_name, params)
