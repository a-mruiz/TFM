# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '6f36562b-45ee-4244-be1e-2c6bf59d8086'
resource_group = 'TFM'
workspace_name = 'TFM_azureml'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='middlebury_1')
dataset.download(target_path='.', overwrite=False)