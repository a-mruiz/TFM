
import json
import azureml.core
from azureml.core import Workspace, Dataset, Environment, Experiment, ScriptRunConfig,Run
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data import FileDataset
import os
import shutil

# from azureml.core.authentication import ServicePrincipalAuthentication

# sp = ServicePrincipalAuthentication(tenant_id="your-tenant-id", # tenantID
#                                     service_principal_id="your-client-id", # clientId
#                                     service_principal_password="your-client-secret") # clientSecret 


# from azureml.core.authentication import AzureCliAuthentication

# cli_auth = AzureCliAuthentication()


use_dataAug=True

# Experiment folder name
experiment_folder = 'Test_azure_training'

# Experiment number
exp_number = '1'


# Load the workspace from the saved config file
ws = Workspace.from_config()


print('Ready to use Azure ML {} to work with {}'.format(
    azureml.core.VERSION, ws.name))
compute_instance_name="tfm-compute-instance"

#obtaining the dataset that was previously created
dataset=FileDataset.get_by_name(ws,"middlebury_1")

#This dataset_input will be passed as argument to the train script (located in the docker container that performs training)
#and will mount the data in the destination
dataset_input = dataset.as_mount()

"""
In the training script, we need to add the following to retrieve the mount point of the dataset->

# The mount point can be retrieved from argument values
import sys
mount_point = sys.argv[1]

# The mount point can also be retrieved from input_datasets of the run context.
from azureml.core import Run
mount_point = Run.get_context().input_datasets['input_1']

"""

# Create a folder for the experiment files
try: 
    os.makedirs(experiment_folder, exist_ok=False)
    print(experiment_folder, 'folder created')
except:
    print("\nDirectory already existed, removing and creating new one with the most updated version...")
    shutil.rmtree(experiment_folder)
    os.makedirs(experiment_folder, exist_ok=True)
    print(experiment_folder, 'folder created')

try:
    ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*",
                                            "*build", "*.fuse*", "*_drive_*", "data_middlebury",
                                            "external_models", "outputs","runs","slurm-*")
    # Copy the necessary Python files into the experiment folder
    shutil.copytree('../Depth_inpainting', experiment_folder+"/Depth_inpainting", ignore=ignore_hidden)
except:
    pass

# Create a Python environment for the experiment
pytorch_env = Environment("environment_pytorch_kornia_full")
# Let Azure ML manage dependencies
pytorch_env.python.user_managed_dependencies = False
# Create a set of package dependencies (conda or pip as required)
pytorch_packages = CondaDependencies.create(conda_packages=['ipykernel', 'matplotlib', 'numpy', 'pillow', 'pip'],
                                            pip_packages=['azureml-sdk', 'torch', 'tqdm', 'tensorboard', 'torchvision', 'loss-landscapes', 'imageio',
                                                          'filelock', 'kornia', 'log_utils', 'net_utils', 'networks', 'numpy',
                                                          'opencv_python_headless', 'Pillow', 'pytorch_msssim', 'ray',
                                                          'scikit_learn','pynvml', 'azure-storage-blob','git+https://github.com/fraunhoferhhi/DeepCABAC.git'])

# Add the dependencies to the environment
pytorch_env.python.conda_dependencies = pytorch_packages
print(pytorch_env.name, 'defined.')

# Register the environment
pytorch_env.register(workspace=ws)

with open('config_blob.json') as json_file:
    config_blob = json.load(json_file)



# Creates the configuration script that will manage environment, compute-instance and pass variables to
# the training script
script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script="Depth_inpainting/main_train_new.py",
                                arguments=[dataset_input,use_dataAug,config_blob["AZURE_STORAGE_CONNECTION_STRING"]],
                                #arguments=[
                                #  '--dataset_input',dataset_input,
                                #  '--dataAug',use_dataAug,
                                #  '--az_blob_conn_str', config_blob["AZURE_STORAGE_CONNECTION_STRING"] 
                                #],
                                environment=pytorch_env, 
                                compute_target=compute_instance_name
                                )


# submit the experiment
experiment = Experiment(workspace=ws, name="Experiment_1")
run = experiment.submit(config=script_config)
#run.wait_for_completion(show_output=True)
