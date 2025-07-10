# import numpy as np
# import cv2

# def disparity_to_depth(baseline, f, img):
#     """This is used to compute the depth values from the disparity map"""

#     # Assumption image intensities are disparity values (x-x') 
#     depth_map = np.zeros((img.shape[0], img.shape[1]))
#     depth_array = np.zeros((img.shape[0], img.shape[1]))

#     for i in range(depth_map.shape[0]):
#         for j in range(depth_map.shape[1]):
#             depth_map[i][j] = 1/img[i][j]
#             depth_array[i][j] = baseline*f/img[i][j]
#             # if math.isinf(depth_map[i][j]):
#             #     depth_map[i][j] = 1

#     return depth_map, depth_array


import numpy as np
import cv2

# def disparity_to_depth(baseline, f, disparity_map):
#     """Vectorized depth computation with correct formula"""
    
#     # Avoid division by zero
#     disparity_safe = np.where(disparity_map > 0, disparity_map, 0.1)
    
#     # Correct depth formula: depth = (baseline * focal_length) / disparity
#     depth_array = (baseline * f) / disparity_safe
    
#     # Create normalized depth map for visualization
#     depth_map = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
#     depth_map = np.uint8(depth_map)
    
#     return depth_map, depth_array
def disparity_to_depth(baseline, f, disparity_map):
    """Safe and accurate depth computation with masking"""

    # Create mask for valid disparity values
    valid_mask = disparity_map > 0

    # Allocate depth array
    depth_array = np.zeros_like(disparity_map, dtype=np.float32)

    # Compute only for valid pixels
    depth_array[valid_mask] = (baseline * f) / disparity_map[valid_mask]

    # Optional: clip depth to remove extreme outliers
    depth_array = np.clip(depth_array, 0.1, 100.0)

    # Normalize only the valid depth range for visualization
    depth_vis = np.zeros_like(depth_array)
    depth_vis[valid_mask] = depth_array[valid_mask]

    # Normalize safely
    if np.any(valid_mask):
        depth_map = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)
    else:
        depth_map = np.zeros_like(disparity_map, dtype=np.uint8)

    return depth_map, depth_array
