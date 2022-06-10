import json
import helper
from model import models 
import time
import torch


cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("===> Using '{}' for computation.".format(device))


compression_params={
    "interv":0.01,
    "stepsize":2**(-0.5*10),
    "stepsize_other":2**(-0.5*19),
    "_lambda": 0.1
}


model=models.BasicModelDeep().to(device)
model=models.Pix2PixGanGenerator().to(device)

helper.compress_model_weights("weights/generator.pt",compression_params,"weights/generator.bin")


model_compressed,_=helper.decode_model_weights(model,'weights/generator.bin')    

helper.test_model(model_compressed,device)
model.load_state_dict(torch.load("weights/generator.pt", map_location=device))
helper.test_model(model,device," normal ")





"""

compression_params={
    "interv":0.01,
    "stepsize":2**(-0.5*10),
    "stepsize_other":2**(-0.5*19),
    "_lambda": 0.5
}


model=models.SelfAttentionCBAM().to(device)


helper.compress_model_weights("weights/weights.pt",compression_params,"weights/weights.bin")


model_compressed,_=helper.decode_model_weights(model,'weights/weights.bin')    

helper.test_model(model_compressed,device)
model.load_state_dict(torch.load("weights/weights.pt", map_location=device))
helper.test_model(model,device," normal ")

"""

