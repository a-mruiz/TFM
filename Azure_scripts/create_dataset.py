import azureml.core
from azureml.core import Workspace, Dataset
from azureml.data.datapath import DataPath

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Get the default datastore
default_ds = ws.get_default_datastore()

dataset=Dataset.File.upload_directory(src_dir="data_middlebury",
                                      target=DataPath(default_ds,"data_middlebury"),
                                      show_progress=True)
dataset = dataset.register(workspace=ws,
                                 name='middlebury_1',
                                 description='Data from middlebury')