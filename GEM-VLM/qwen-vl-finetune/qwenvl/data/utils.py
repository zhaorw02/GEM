import os  
import torch
import numpy as np
from PIL import Image
import json
import matplotlib

path_map = {
    "gem": "your_path_to_gem_data_depth"  # TODO: set your path to depth of gem data
    
  
   
}
def get_depth_path(source, video_metadatas=None):
    return_path_list = []
    
    # TODO: support image depth loading 
    data_path = path_map[source["data_source"]]
    image_path = source['image'] 
    dir_name, base_name = os.path.split(image_path)
    depth_path = os.path.join(data_path, "depth")
    depth_image_path = os.path.join(depth_path, base_name)
    if os.path.exists(depth_image_path):
        return_path_list.append(depth_image_path)
    else:
        print(f"Depth image {depth_image_path} not found for {image_path}")
    
    return return_path_list



def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    ret_type=np.uint8,
    cmap="Spectral",
):
    """
    Visualize a depth map using a colormap.

    Args:
        depth: Input depth map array
        depth_min: Minimum depth value for normalization. If None, uses percentile
        depth_max: Maximum depth value for normalization. If None, uses percentile
        percentile: Percentile for min/max computation if not provided
        ret_minmax: Whether to return min/max depth values
        ret_type: Return array type (uint8 or float)
        cmap: Matplotlib colormap name to use

    Returns:
        Colored depth visualization as numpy array
        If ret_minmax=True, also returns depth_min and depth_max
    """
    depth = depth.copy()
    depth.copy()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]
    if depth_min is None:
        if valid_mask.sum() <= 10:
            depth_min = 0
        else:
            depth_min = np.percentile(depth[valid_mask], percentile)
    if depth_max is None:
        if valid_mask.sum() <= 10:
            depth_max = 0
        else:
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth = 1 - depth
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    if ret_type == np.uint8:
        img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    elif ret_type == np.float32 or ret_type == np.float64:
        img_colored_np = img_colored_np[0]
    else:
        raise ValueError(f"Invalid return type: {ret_type}")
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np


    
def load_depth_color_image(path, target_size=None, visualize_color=True):
    # 16-bit PNG depth image needs visualization to be converted to 3-channel RGB for better representation
    try:
  
        depth_img = Image.open(path)
        if target_size:
            depth_img = depth_img.resize(target_size, Image.NEAREST)
      
        depth_np = np.array(depth_img).astype(np.float32)
        if visualize_color:
            depth_np = visualize_depth(depth_np)
        else:
            depth_np = depth_np.astype(np.uint8) 
        depth_rgb = torch.from_numpy(depth_np).permute(2, 0, 1)
        return depth_rgb

    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None
        


