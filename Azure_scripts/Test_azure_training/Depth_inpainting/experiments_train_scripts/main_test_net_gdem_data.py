"""
This file contains the main logic to train the model
Author:Alejandro
"""

from dataloaders.JaimeHoviDataloaderFile import HoviDataLoader
import torch
from dataloaders.MiddleburyDataloaderFile import MiddleburyDataLoader
from dataloaders.GdemDataloaderFile import GdemDataLoader
from helpers.helper import LRScheduler, Logger, save_result_row, save_test_result
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
from helpers import graph_utils

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
    
    jaime_or_gdem="jaime"
    
    
    train_or_test = "test"
    deactivate_logger=True
    
    model = models.InceptionAndAttentionModel()
    checkpoint = torch.load(
            "model_best.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    """#2. Dataloaders"""
    if jaime_or_gdem=="gdem":
        dataset_test = GdemDataLoader('test')
        out_folder="outputs/test_gdem/inceptionandatt/"
    if jaime_or_gdem=="jaime":
        dataset_test = HoviDataLoader("test")
        out_folder="outputs/test_hovi/inceptionandatt/"
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        sampler=None)

    psnr_val=[]
    loss_val=[]
    with torch.no_grad():
        model.eval()
        for i,batch_data in enumerate(test_dataloader):
            batch_data = {
                key: val.to(device) for key, val in batch_data.items() if val is not None
            }
                    
            output=model(batch_data)
            val_current_psnr = -losses.psnr_loss(output, batch_data['gt']).item()
            psnr_val.append(val_current_psnr)
            save_test_result(batch_data, output, "out_"+str(i)+".png", folder=out_folder)
            
    graph_utils.make_graph([psnr_val], ['val'], range(0,len(psnr_val)), title="PSNR values validation on gdem data", x_label="epochs",
        y_label="Loss", x_lim_low=0, x_lim_high=len(psnr_val), show=False, subtitle="", output_dir=out_folder,x_ticker=5,y_ticker=1.5)
    print("End of testing process!")
if __name__ == '__main__':
    main()