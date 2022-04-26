"""
This file contains the main logic to train the model
Author:Alejandro
"""

import torch
from dataloaders.MiddleburyDataloaderFile import MiddleburyDataLoader
from helpers.helper import LRScheduler, Logger
import helpers.losses as losses
import model.models as models
from dataloaders.BoschDataloaderFile import BoschDataLoader
import time
from tqdm import tqdm
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from train_test_routines.train_routine import train_model, test_model
from model.external import SelfSup_FangChang_2018, PENet_2021, TWISE_2021
import sys

sys.path.append("/media/beegfs/home/t588/t588188/.local/lib/python3.9/site-packages") 
cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("===> Using '{}' for computation.".format(device))



def main():
    
    train_or_test = "train"

    print("===> Starting framework...")
    """Some parameters for the model"""
    lr = 0.01
    weight_decay = 1e-06
    epochs = 15
    params = {"mode": train_or_test, "lr": lr,
              "weight_decay": weight_decay, "epochs": epochs,
              "bs":1}
    
    """#1. Load the model"""
    print("===> Loading model...")
    
    model = models.DummyModel()
    model_option = "DummyModel"
    params['model']=model_option
    
    writer = SummaryWriter(comment=f'')

    #info = nvmlDeviceGetMemoryInfo(h)
    #print(f'total    : {info.total// 1024 ** 2}')
    #print(f'free     : {info.free// 1024 ** 2}')
    #print(f'used     : {info.used// 1024 ** 2}')


    if torch.cuda.is_available():
        model.cuda()
    """#2. Dataloaders"""
    print("===> Configuring Dataloaders...")
    #option = int(
    #    input("Which dataset do you want to use?:\n\t-1=Bosch\n\t-2=GDEM\n"))

    dataset = MiddleburyDataLoader('train', augment=True, preprocess_depth=False)
    dataset_test = MiddleburyDataLoader('test', augment=False, preprocess_depth=False)
    dataset_option = "Middlebury"

    params['dataset_option'] = dataset_option
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['bs'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        sampler=None)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=None)

    # If needed to run in multiple GPUs, use--->DistributedDataParallel


    """#3. Create optimizer and criterion"""
    print("===> Creating optimizer and criterion...")
    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(
        model_named_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))


    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = losses.CombinedLoss()

    """#4. If no errors-> Create Logger object to backup code and log training"""
    logger = Logger(params)

    
    lr_scheduler=LRScheduler(optimizer)
    
    """#5. Split execution between train and test"""
    if train_or_test == "train":
        train_model(model, epochs, params, optimizer,
                    logger, train_dataloader,test_dataloader, criterion, device,lr_scheduler,writer)
        logger.generateTrainingGraphs()
        #torch.save(model.state_dict(), "model_last.pth")
    elif train_or_test == "test":
        avg_loss = test_model(model, test_dataloader,
                              criterion, logger, device)

        print("Test avg loss--->"+str(avg_loss))
    

if __name__ == '__main__':
    main()