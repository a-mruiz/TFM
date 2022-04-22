"""
This file contains the main logic to train the model
Author:Alejandro
"""

import torch
from helpers.helper import Logger
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
    option = int(
        input("Do you want to train or test the model(1/2)?:\n\t-1=Train\n\t-2=Test\n"))

    if option == 1:
        train_or_test = "train"
    else:
        train_or_test = "test"

    print("===> Starting framework...")
    """Some parameters for the model"""
    lr = 0.09
    weight_decay = 1e-05
    epochs = 5
    params = {"mode": train_or_test, "lr": lr,
              "weight_decay": weight_decay, "epochs": epochs}

    """#1. Load the model"""
    print("===> Loading model...")
    option = int(
        input("Which model do you want to use?:\n\t-1=Dummy\n\t-2=Basic\n\t-3=BasicDeep\n\t-4=TwoBranchModel\n\t-5=FancgChang_SelfSupervised_2018\n\t-6=PENet_2021\n\t-7=TWISE_2021\n\t-8=Unsupervised_Backprojection_2021"))

    if option == 1:
        model = models.DummyModel()
        model_option = "Dummy"
    elif option == 2:
        model = models.BasicModel()
        model_option = "Basic"
    elif option == 3:
        model = models.BasicModelDeep()
        model_option = "BasicDeep"
    elif option == 4:
        model = models.TwoBranchModel()
        model_option = "TwoBranch"
    elif option == 5:
        model = SelfSup_FangChang_2018.DepthCompletionNet()
        if train_or_test == "test":
            checkpoint = torch.load(
                "external_models/Self-supervised Sparse to dense/model_best.pth.tar", map_location=device)
            model.load_state_dict(checkpoint['model'])
        model_option = "SelfSup_FangChang_2018"
    elif option == 6:
        model = PENet_2021.PENet_C2()
        if train_or_test == "test":
            checkpoint = torch.load(
                "external_models/PENet: Precise and Efficient Depth Completion/pe.pth.tar", map_location=device)
            model.load_state_dict(checkpoint['model'])
        model_option = "PENet_2021"
    elif option == 7:
        model = TWISE_2021.network()
        if train_or_test == "test":
            checkpoint = torch.load(
                "external_models/Twin-Surface-Extrapolation-TWISE/TWISE_gamma2.5/model_best.pth.tar", map_location=device)
            model.load_state_dict(checkpoint['model'])
        model_option = "TWISE_2021"
    elif option == 8:
        model = Unsupervised_Backprojection_2021.KBNetModel(
            input_channels_image=3,
            input_channels_depth=1,
            min_pool_sizes_sparse_to_dense_pool=[5, 7, 9, 11, 13],
            max_pool_sizes_sparse_to_dense_pool=[15, 17],
            n_convolution_sparse_to_dense_pool=3,
            n_filter_sparse_to_dense_pool=8,
            n_filters_encoder_image=[48, 96, 192, 384, 384],
            n_filters_encoder_depth=[16, 32, 64, 128, 128],
            resolutions_backprojection=[0, 1, 2, 3],
            n_filters_decoder=[256, 128, 128, 64, 12],
            deconv_type='up',
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            min_predict_depth=0,
            max_predict_depth=100,
            device=device
        )
        if train_or_test == "test":
            checkpoint = torch.load(
                "external_models/Unsupervised Depth Completion with Calibrated Backprojection Layers/pretrained_models/void/kbnet-void1500.pth", map_location=device)
            model.load_state_dict(checkpoint['model'])
        model_option = "Unsupervised_Backprojection_2021"
    else:
        raise(RuntimeError("ALEJANDRO ERROR--> SELECT A CORRECT VALUE"))
    params['model_option'] = model_option
    """#2. Dataloaders"""
    print("===> Configuring Dataloaders...")
    option = int(
        input("Which dataset do you want to use?:\n\t-1=Bosch\n\t-2=GDEM\n"))

    if option == 1:
        dataset = BoschDataLoader('train')
        dataset_test = BoschDataLoader('test')
        dataset_option = "Bosch"
    elif option == 2:
        dataset = BoschDataLoader('train')
        dataset_test = BoschDataLoader('test')
        dataset_option = "GDEM"
    else:
        raise(RuntimeError("ALEJANDRO ERROR--> SELECT A CORRECT VALUE"))
    params['dataset_option'] = dataset_option
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=4,
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
    criterion = losses.RMSELoss()

    """#4. If no errors-> Create Logger object to backup code and log training"""
    logger = Logger(params)

    """#5. Split execution between train and test"""
    if train_or_test == "train":
        train_model(model, epochs, params, optimizer,
                    logger, train_dataloader, criterion, device)
    elif train_or_test == "test":
        avg_loss = test_model(model, test_dataloader,
                              criterion, logger, device)
        print("Test avg loss--->"+str(avg_loss))


if __name__ == '__main__':
    main()
