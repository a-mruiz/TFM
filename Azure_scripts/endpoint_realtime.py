import os
import logging
import json
import numpy
import torch
import Test_azure_training.Depth_inpainting.model.models as models
import base64
import numpy as np


def ToTensor(img):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W

    Convert a ``numpy.ndarray`` to tensor.

    Args:
        img (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    def _is_numpy_image(img):
        return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

    if not (_is_numpy_image(img)):
        raise RuntimeError('AlEJANDRO ERROR-->img should be ndarray or dimesions are wrong. Got type (' +
                           str(type(img))+') and dimensions ('+str(img.shape)+')')

    if isinstance(img, np.ndarray):
        # handle numpy array
        if img.ndim == 3:
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
        elif img.ndim == 2:
            img = torch.from_numpy(img.copy())
        else:
            raise RuntimeError('AlEJANDRO ERROR-->img should be ndarray or dimesions are wrong. Got type (' +
                               str(type(img))+') and dimensions ('+str(img.shape)+')')
        return img





def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global device
    global model

    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        #cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info("===> Using '{}' for computation.".format(device))
    #loading the model
    model_path = "Test_azure_training/Depth_inpainting/model_best.pt"
    model = models.SelfAttentionCBAM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    logging.info("Init complete")


def run(json_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    """
    #Deserializar las imagenes que vienen en JSON
    #Pasar las imágense a tensores
    #Inferencia
    #Crear JSON con la imágen de respuesta (salida de la red)
    #En caso de conocer el gt, calcular loss y devolver estadísticas

    """
    For encoding b64->base64.b64encode(img)
    For decoding b64->base64.b64decode(img)
    """
    
    logging.info("Request received")
    
    json_data = json.loads(json_data)
    rgb_img = base64.b64decode(json_data['rgb'])
    depth_img = base64.b64decode(json_data['d'])
    
    rgb=ToTensor(rgb_img).float()/255
    depth=ToTensor(depth_img).float()/255
    
    
    #Data to feed the model
    items={"rgb":rgb.to(device), "d":depth.to(device)}
    
    output = model(items)
    
    
    

    
    
    
    logging.info("Request processed")
    return True