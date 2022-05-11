import logging
import json
import torch
import model as models
import base64
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

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

cmap = plt.cm.binary

def save_image_color(img_merge, filename):
    cv2.imwrite(filename, img_merge)

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    
    return depth.astype('uint8')  


def save_result_row(rgb,output, depth, name="test.png"):

    #unorm_rgb = transforms.Normalize(mean=[-0.4409/0.2676, -0.4570/0.2132, -0.3751/0.2345],
    #                         std=[1/0.2676, 1/0.2132, 1/0.2345])
    #unorm_d = transforms.Normalize(mean=[-0.2674/0.1949],
    #                         std=[1/0.1949])
    #unorm_gt = transforms.Normalize(mean=[-0.3073/0.1761],
    #                         std=[1/0.1761])

    unorm_rgb=transforms.Normalize(mean=[0,0,0],std=[1,1,1])
    unorm_d=transforms.Normalize(mean=[0],std=[1])
    #unorm_gt=transforms.Normalize(mean=[0],std=[1])

    rgb=unorm_rgb(torch.squeeze(rgb,0))
    depth=unorm_d(torch.squeeze(depth,0))
    #gt=unorm_gt(depth)
    output=unorm_d(torch.squeeze(output,0))
    #depth=unorm_d(batch_data['d'])

    #rgb=batch_data['rgb'][0,...]
    #depth=batch_data['d']
    #gt=batch_data['gt']



    #print("OUTPUT Size------------------->"+str(output.shape))
    img_list=[]
    
    rgb = np.squeeze(rgb.data.cpu().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))*255
    img_list.append(rgb)

    depth = depth_colorize(np.squeeze(depth.data.cpu().numpy()))
    img_list.append(depth)

    output = depth_colorize(np.squeeze(output.data.cpu().numpy()))

    img_list.append(output)
    
    img_merge = np.hstack([img_list[0], img_list[1], img_list[2]])
    img_merge= img_merge.astype('uint8')
    
    return img_merge
    #print("saving img to "+str(folder+name))




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
    model_path = "model_best.pt"
    model = models.SelfAttentionCBAM()
    state_dict=torch.load(model_path, map_location=device)
    logging.info(state_dict.keys())
    model.load_state_dict(state_dict)
    
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
        
    rgb_np = np.frombuffer(rgb_img, dtype='uint8')
    rgb_img = cv2.imdecode(rgb_np, 1)
    depth_np = np.frombuffer(depth_img, dtype='uint8')
    depth_img = cv2.imdecode(depth_np, 1)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    
    
    rgb=ToTensor(rgb_img).float()/255
    depth=ToTensor(depth_img).float()/255
    
    norm_rgb=transforms.Normalize(mean=[0,0,0],std=[1,1,1])
    norm_d=transforms.Normalize(mean=[0],std=[1])
    
    depth=torch.unsqueeze(depth,0)
    
    rgb=norm_rgb(rgb)
    depth=norm_d(depth)
    
    rgb=torch.unsqueeze(rgb,0)
 
    depth=torch.unsqueeze(depth,0)
    print(rgb.shape)
    print(depth.shape)
    #Data to feed the model
    items={"rgb":rgb.to(device), "d":depth.to(device),"g":None}
    
    output = model(items)
        
    #print(type(output))
    #output=torch.squeeze(output,0)
    #output=transforms.ToPILImage()(output)
    #print(type(output))
    #output_np=np.array(output)
    
    
    merged_result=save_result_row(rgb,output,depth)
    
    save_image_color(merged_result,"test.png")
    
    
    
    
    #output.save("test.png")

    logging.info("Request processed")
    return True