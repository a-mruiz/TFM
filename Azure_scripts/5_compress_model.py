
import azureml.core
from azureml.core import Workspace, Dataset, Environment, Experiment, ScriptRunConfig,Run
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data import FileDataset
import os
import shutil


# Experiment number
exp_number = '2'


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(
    azureml.core.VERSION, ws.name))
compute_instance_name="tfm-compute-instance"



# Create a Python environment for the experiment
pytorch_env = Environment("environment_pytorch_kornia_full")
# Let Azure ML manage dependencies
pytorch_env.python.user_managed_dependencies = False
# Create a set of package dependencies (conda or pip as required)
pytorch_packages = CondaDependencies.create(conda_packages=['ipykernel', 'matplotlib', 'numpy', 'pillow', 'pip'],
                                            pip_packages=['azureml-sdk', 'torch', 'tqdm', 'tensorboard', 'torchvision', 'loss-landscapes', 'imageio',
                                                          'filelock', 'kornia', 'log_utils', 'net_utils', 'networks', 'numpy',
                                                          'opencv_python_headless', 'Pillow', 'pytorch_msssim', 'ray',
                                                          'scikit_learn','pynvml', 'azure-storage-blob'])

# Add the dependencies to the environment
pytorch_env.python.conda_dependencies = pytorch_packages
print(pytorch_env.name, 'defined.')

# Register the environment
pytorch_env.register(workspace=ws)


#*############################
# * USE ENVIRONMENT TO RUN
# * A SCRIPT AS AN EXPERIMENT IN
# * A COMPUTE CLUSTER
# *

# Create a script config (Uses docker to host environment)
# Using 'as_download' causes the files in the file dataset to be downloaded to
# a temporary location on the compute where the script is being run.
# Reference to datasets and the paths where they will be downloaded in the environment
script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script="Depth_inpainting/main_train_new.py",
                                arguments=[dataset_input,use_dataAug],
                                #arguments=['--gt-data', gt_ds.as_named_input('gtMaps_data').as_download(),
                                #           '--preProcessed-data', preProcessed_ds.as_named_input(
                                #               'preProcessed_data').as_download(),
                                #           '--patients_list_train', patients_list_train,
                                #           '--patient_test', patient_test,
                                #           '--batch_dim', batch_dim,
                                #           '--epochs', epochs,
                                #           '--batch_size', batch_size,
                                #           '--patch_size', patch_size,
                                #           '--k_folds', k_folds,
                                #           '--learning_rate', lr,
                                #           '--model_name', model_name
                                #           ],
                                environment=pytorch_env, 
                                compute_target=compute_instance_name
                                )

#*################################
# * SUBMIT THE EXPERIMENT TO AZURE
# *
# submit the experiment
experiment = Experiment(workspace=ws, name="Experiment_1")
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
