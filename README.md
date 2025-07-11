# Stereo Vision Depth Estimation

This project implements a complete stereo vision pipeline for depth estimation using stereo image pairs. It includes camera calibration validation, stereo rectification, disparity map generation, depth computation, and YOLOv5-based object detection with depth annotation.

## 📁 Project Structure

```
DEPTH TRIAL/
├── Dataset1/                  # Stereo image pair 1
├── Dataset2/                  # Stereo image pair 2
├── Dataset3/                  # Stereo image pair 3
├── calibration.py             # Computes Fundamental & Essential matrices and camera pose
├── corresModify.py            # Keypoint detection and matching
├── depth.py                   # Disparity and depth map calculation
├── mainnotebook.ipynb         # Jupyter notebook version of main flow
├── rectification.py           # Stereo image rectification
├── requirements.txt           # Python dependencies
├── yolov5s.pt                 # Pretrained YOLOv5 model
```

## 🚀 Features

- **Feature Matching** using ORB
- **Fundamental & Essential Matrix** estimation with RANSAC
- **Camera Pose Extraction** from Essential Matrix
- **Stereo Rectification** using OpenCV’s `stereoRectifyUncalibrated`
- **Disparity Map** using StereoBM
- **Depth Map** computed via triangulation formula
- **YOLOv5** object detection with average depth overlay

## 🔧 Getting Started

### ✅ Prerequisites

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Torch & torchvision
- YOLOv5 (via `torch.hub` or local .pt file)

Install dependencies:

```bash
pip install -r requirements.txt
```

### ▶️ Run the Project

```bash
python mainnotebook.ipynb
```


Choose a dataset when prompted (1/2/3) to visualize rectification, disparity, and depth.

## 📌 Notes

- The stereo images are assumed to be from a **calibrated setup**.
- The calibration pipeline is included to validate epipolar geometry but is not directly used for triangulation.
- Rectification is performed using fundamental matrix and matched keypoints.

## 💡 Future Enhancements

- Replace StereoBM with StereoSGBM or deep networks like FastDepth
- Integrate real-time webcam support for dynamic depth estimation
- Improve depth accuracy by refining calibration (with distortion coefficients)
- Fuse object and depth data for robotics or AR applications

## 👨‍💻 Author

**Rohith Sunil**  
July 2025
