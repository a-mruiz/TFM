# Azure scripts

A file named `config.json` must be present in the directory. It contains the azure connection strings and values to successfully connect to Azure Machine Learning Studio. This file is downloaded from AzureML in the workspace section.

Another file named `config_blob.json` must also be present if needed to connect with the blobs and see the different information. it contains one value `AZURE_STORAGE_CONNECTION_STRING` to connect to blob.

## Scripts

* In order to test connection with Azure, run `test_azure_connection.py`
* In order to test connection with Azure Blobs, run `test_blobs.py`
* In order to create a dataset in AzureML, run `create_dataset.py`. It will upload the local directory and create a new dataset object available in AzureML.
* In order to create a training experiment and run it in AzureML, run `deploy_train.py`. Don't forget to have the compute-instance powered on!!