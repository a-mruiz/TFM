"""
Just used to check that access with AzureML cloud is working properly

"""

import azureml.core
from azureml.core import Workspace, Experiment

#*############################
#* CHECK AZUREML-CORE VERSION
#*
print("Ready to use Azure ML", azureml.core.VERSION)


#*##################################################
#* CONNECT TO YOUR AZURE MACHINE LEARNING WORKSPACE
#*
# Store first the workspace connection information in a JSON.
# This can be downloaded from the Azure portal or the workspace details
# pane in Azure Machine Learning studio.
ws = Workspace.from_config('config.json')
print(ws.name, "loaded")

#*#####################################
#* PRINT AVAILABLE RESOURCES FROM YOUR 
#* AZURE MACHINE LEARNING WORKSPACE
#*
print("Compute Resources:")
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print("\t", compute.name, ':', compute.type)