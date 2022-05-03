from cv2 import mean
from ip_basic_depth_map_utils import fill_in_multiscale
import torch
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

cmap = plt.cm.Greys

def save_result_row(depth, validity_map_depth,depth_validity, depth_wout_outliers, name):
    """Will save a row with the different images rgb+depth+gt+output

    Args:
        batch_data ([type]): [description]
        output ([type]): [description]
    """
    #print("OUTPUT Size------------------->"+str(output.shape))
    img_list=[]

    depth = depth_colorize(np.squeeze(depth.cpu().numpy()))
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)
    
    depth = depth_colorize(np.squeeze(validity_map_depth.cpu().numpy()))
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)
    
    depth = depth_colorize(np.squeeze(depth_validity.cpu().numpy()))
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)

    depth = depth_colorize(np.squeeze(depth_wout_outliers.cpu().numpy()))
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)
    
    img_merge = np.hstack(img_list)
    img_merge= img_merge.astype('uint8')
    save_image_color(img_merge, name)

def save_result_row_2(depth, validity_map_depth,depth_validity, depth_wout_outliers, name):
    """Will save a row with the different images rgb+depth+gt+output

    Args:
        batch_data ([type]): [description]
        output ([type]): [description]
    """
    #print("OUTPUT Size------------------->"+str(output.shape))
    img_list=[]

    depth = depth_colorize(depth)
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)
    
    depth = depth_colorize(validity_map_depth)
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)
    
    depth = depth_colorize(depth_validity)
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)

    depth = depth_colorize(depth_wout_outliers)
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)
    
    img_merge = np.hstack(img_list)
    img_merge= img_merge.astype('uint8')
    save_image_color(img_merge, name)



def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


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


def save_image_color(img_merge, filename):
    cv2.imwrite(filename, img_merge)

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    
    return depth.astype('uint8')   


def read_rgb(path):
    rgb=cv2.imread(path)
    #print("SHAPE JUST READED RGGB----->"+str(rgb.shape))
    gray = np.array(Image.fromarray(rgb).convert('L'))
    gray = np.expand_dims(gray, -1)
    #print(rgb.shape)
    #print("RGB->"+str(rgb.shape))
    return rgb, gray


def read_depth(path):
    depth=cv2.imread(path)
    depth = np.expand_dims(np.array(Image.fromarray(depth).convert('L')),2)
    #depth=np.expand_dims(np.load(path)['arr_0'],2)
    #print(depth.shape)
    #print("DEPTH->"+str(depth.shape))
    return depth


#depth=read_depth("data_gdem/test/sparse/v0.png")

#sparse_depth=ToTensor(sparse_depth).float()
#
##sparse_depth=1-sparse_depth
#
#outlier_removal=OutlierRemoval(3, 5)
#depth_infuser=DepthInfuser()
#
#
#print(sparse_depth.min())
## Validity map is where sparse depth is available
#validity_map_depth = torch.where(
#                    sparse_depth ==0,
#                    torch.ones_like(sparse_depth)*255,
#                    sparse_depth)
#
##filtered_depth, depth_val= outlier_removal.remove_outliers(sparse_depth=sparse_depth, validity_map=validity_map_depth)
#
#filtered_depth, depth_val= depth_infuser.infuse_depth(sparse_depth=sparse_depth, validity_map=validity_map_depth)
#
##depth = depth_colorize(np.squeeze(filtered_depth.cpu().numpy()))
#
#save_result_row(sparse_depth,validity_map_depth,depth_val,filtered_depth , "testing.png")

# 7x7 cross kernel
kernel_cross_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
kernel_cross_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
kernel_diamond_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
# 3x3 cross kernel
kernel_cross_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

kernel = np.ones((3,3),np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)
kernel_7 = np.ones((7, 7), np.uint8)
kernel_11 = np.ones((11,11), np.uint8)


#depth_closing = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel_cross_7)
#depth_closing_2 = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)


#save_result_row_2(np.squeeze(depth),depth_closing,depth_closing_2,np.squeeze(depth)-depth_closing_2, "testing_2.png")


def quick_remove_occlusions(depth):
    """Quickly dilatate the image to remove some occlusions

    Args:
        depth ([type]): [Depth map to remove occlusions]
    """
    #print("Depth Max->"+str(depth.max()))
    #print("Depth Min->"+str(depth.min()))
    #print("Depth Mean->"+str(depth.mean()))
    depth_2=depth.copy()
    depth_closing_2 = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)

    depth_2 = cv2.morphologyEx(depth_2, cv2.MORPH_CLOSE, kernel)

    # Dilate
    depth_map = cv2.dilate(depth_2, kernel_diamond_5)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, kernel_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    #save_result_row_2(np.squeeze(depth),depth_map,depth_2,np.squeeze(depth)-depth_closing_2, "testing_3.png")
    return depth_map


def slow_remove_occlusions(depth):
    """Will remove occlusions at different ranges, it will use different kernels for different depths.

    Args:
        depth ([type]): [description]
    """
    #print("Depth Max->"+str(depth.max()))
    #print("Depth Min->"+str(depth.min()))
    #print("Depth Mean->"+str(depth.mean()))
    
    max_depth=depth.max()
    min_depth=depth.min()
    mean_depth=depth.mean()
    
    depth_2=depth.copy()
    #depth=1-depth
    
    limit_near=mean_depth/1.5
    limit_medium=mean_depth*1.5
        
    # Calculate the masks for pixels near, medium and far
    pixels_far = (depth > 0) & (depth <= limit_near)
    pixels_med = (depth > limit_near) & (depth<=limit_medium)
    pixels_near = (depth>limit_medium)
    
    # Dilate depending on depths
    depth_map_far = cv2.dilate(np.multiply(depth, pixels_far), kernel_cross_3) 
    depth_map_med = cv2.dilate(np.multiply(depth, pixels_med), kernel_cross_5)   
    depth_map_near = cv2.dilate(np.multiply(depth, pixels_near), kernel_cross_7)   

    valid_pixels_near = (depth_map_near > 0.1)
    valid_pixels_med = (depth_map_med > 0.1)
    valid_pixels_far = (depth_map_far > 0.1)
    
    #print("Depth Max->"+str(depth_map_near.max()))
    #print("Depth Min->"+str(depth_map_near.min()))
    #print("Depth Mean->"+str(depth_map_near.mean()))

    depth=np.squeeze(depth)
    
    # Combine dilated versions, starting farthest to nearest
    depth[valid_pixels_far] = depth_map_far[valid_pixels_far]
    depth[valid_pixels_med] = depth_map_med[valid_pixels_med]
    depth[valid_pixels_near] = depth_map_near[valid_pixels_near]
     
    depth_map_closed = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel_7)
    depth_map_closed_2 = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel_cross_7)

    depth_map_opened = cv2.morphologyEx(depth_map_closed, cv2.MORPH_OPEN, kernel_11)

    depth_map_closed_2, _ = fill_in_multiscale(depth, 255)

    #print("DEPTH MAP SHAPE->"+str(depth_map_closed.shape))
    
    quick_depth=quick_remove_occlusions(depth_2)
        
    #depth=1-depth
    #save_result_row_2(np.squeeze(depth_2),depth_map_closed,quick_depth,depth, "testing_4.png")
    return depth_map_closed




#quick_remove_occlusions(depth)
#slow_remove_occlusions(depth)