import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

try:
    print("Azure Blob Storage v" + __version__ + " - Python quickstart sample")

    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a unique name for the container
    container_name = "conteinertostoremodels"
    try:
        # This will only work the first time, before this container is created
        # Create the container
        container_client = blob_service_client.create_container(container_name)
    except:
        container_client = blob_service_client.get_container_client(container_name)
    print("Retrieving a list of the blobs in the container...")
    # List the blobs in the container
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        print("\t" + blob.name)
    """
    For downloading file->
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    blob_service_client.get_blob_to_path(
        container_name, local_file_name, full_path_to_file2)
    """ 
    """
    For uploading file->
    # Upload the created file, use local_file_name for the blob name
    blob_service_client.create_blob_from_path(
        container_name, local_file_name, full_path_to_file)
    """
    blob_service_client.create_blob_from_path(
        container_name, local_file_name, full_path_to_file)
    
except Exception as ex:
    print('Exception:')
    print(ex)
    