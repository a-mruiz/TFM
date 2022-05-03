"""
Used to upload data from the local directory to the default Azure Datastore in a Blob
"""
from distutils.command.build_scripts import first_line_re
import azureml.core
from azureml.core import Workspace
import os

def upload_files_in_path(path,azure_dir,datastore):
    """Uploads the file in files to azure datastore in the specified azure_dir

    Args:
        path (str): Path to search the files to upload
        azure_dir (str): Azure directory where to store the uploaded files
        datastore (AzureDatastore): Azure datastore where to store the files
    """
    files=os.listdir(path)
    files=[path+file for file in files]
    datastore.upload_files(files=files,
                           target_path=azure_dir,
                           overwrite=True,
                           show_progress=False)
                           #relative_root=path)



# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Get the default datastore
default_ds = ws.get_default_datastore()

# Enumerate all datastores, indicating which is the default
for ds_name in ws.datastores:
    print("\t"+ds_name, "- Default =", ds_name == default_ds.name)

print("Uploading dataset to the default datastore...")

upload_files_in_path("data_middlebury/train/gt/","data_middlebury/train/gt/",default_ds)
upload_files_in_path("data_middlebury/train/rgb/","data_middlebury/train/rgb/",default_ds)
upload_files_in_path("data_middlebury/test/gt/","data_middlebury/test/gt/",default_ds)
upload_files_in_path("data_middlebury/test/rgb/","data_middlebury/test/rgb/",default_ds)

print("Finished uploading the files!")


