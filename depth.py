import numpy as np
import cv2

def disparity_to_depth(baseline, f, disparity_map):
    """Vectorized depth computation with correct formula"""
    
    # Avoid division by zero
    disparity_safe = np.where(disparity_map > 0, disparity_map, 0.1)
    
    # Correct depth formula: depth = (baseline * focal_length) / disparity
    depth_array = (baseline * f) / disparity_safe
    
    # Create normalized depth map for visualization
    depth_map = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)
    
    return depth_map, depth_array

