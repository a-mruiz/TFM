from cv2 import mean
import torch
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

cmap = plt.cm.Greys



import collections

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def fill_in_multiscale(depth_map, max_depth=100.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = (depths_in > 30.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict












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