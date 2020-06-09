# face-reconstruction
3D face reconstruction from front and side images.

## Getting Started

### Prerequisite
* Python 3.6 (numpy, skimage, scipy)

* TensorFlow-gpu = 1.15.0

* keras = 2.3.1

* dlib

* opencv2 

#### Anaconda
This conda environment was able to run demo.py on a windowds computer with GTX 1070 GPU on 09.06.2020.
 ```bash
conda create --name face-recon python=3.6
conda activate face-recon
conda install -c anaconda scipy
conda install -c anaconda scikit-image
conda install -c conda-forge dlib
conda install -c conda-forge opencv
conda install -c anaconda keras-gpu
 ```

### Usage

 1. Clone the repository.
  ```bash
  git clone https://github.com/olalium/face-reconstruction
  cd face-reconstruction
  ```
 
 2. Clone the ICP repository to face-reconstruction folder.
  ```bash
  git clone https://github.com/ClayFlannigan/icp
  ```
 3. Clone face3d repository to face-reconstruction folder.
  ```bash
  git clone https://github.com/YadiraF/face3d
  ```
 4. Download trained model and shape predictor for keypoints (Not for long hopefully..)
 
  Navigate to the ned-data folder
  ```bash
  cd Data/net-data
  ```
  add these models:

  Shape preditctor:  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

  Trained CNN:  https://drive.google.com/file/d/1VJkvxDNIoLUOK1eQ9jZHyu_Xi68xYCSV/view?usp=sharing

 5. Run demo
  ```bash
  cd ../..
  python demo.py
 ```
