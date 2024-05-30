## C++ Implementation for MV-GLMB-AB
- The following functions are implemented in C++ and can be called from Python:
    - `gibbs_multisensor_approx_dprobsample`
    - `MCAdaptiveBirth`, Monte Carlo Adaptive Birth implementation for paper "Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking".
    - `bboxes_ioi_xyah_back2front_all`, compute Intersection over Area (IOA) from objects closest to the camera.
    - `MSGLMB`, C++ class for MV-GLMB-AB filter
    - `meanShift`, clustering algorithm implementation to obtain a cluster of ground plane points that uses to generate new birth targets.
    - `KMeans`
    - `multisensor_lapjv`, measurement association for each sensor using LapJV

### Requirements

- Python 3.7
- C++ compiler (eg. Windows: Visual Studio 15 2017, Ubuntu: g++)
- pybind11 `git submodule add https://github.com/pybind/pybind11.git`
- EigenRand `git submodule add https://github.com/bab2min/EigenRand.git`
- Linear Assignment Problem solver `https://github.com/gatagat/lap.git`
### Install

`python setup.py build develop`

Install Eigen for Windows (after the following steps, add include directory `C:\eigen-3.4.0` for example.)
1) Download Eigen 3.4.0 (NOT lower than this version) from it official website https://eigen.tuxfamily.org/ or [ZIP file here](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip).
2) `mkdir build_dir`
3) `cd build_dir`
4) `cmake ../`
5) `make install`, this step does not require

Install Eigen for Linux
1) [install and use eigen3 on ubuntu 16.04](https://kezunlin.me/post/d97b21ee/) 
2) `sudo apt-get install libeigen3-dev` libeigen3-dev is installed install to `/usr/include/eigen3/` and `/usr/lib/cmake/eigen3`.
3) Thus, we must make a change in **CMakeLists.txt** `SET( EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3" )` to `SET( EIGEN3_INCLUDE_DIR "/usr/include/eigen3/" )`.

### Paper References

- Vo, B. N., Vo, B. T., & Beard, M. (2019). **_Multi-sensor multi-object tracking with the generalized labeled multi-Bernoulli filter_**. IEEE Transactions on Signal Processing, 67(23), 5952-5967.
- Ong, J., Vo, B. T., Vo, B. N., Kim, D. Y., & Nordholm, S. (2020). **_A bayesian filter for multi-view 3D multi-object tracking with occlusion handling_**. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Trezza, A., Bucci, D. J., & Varshney, P. K. (2022). **_Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking_**. IEEE Transactions on Signal Processing, 70, 1010-1025.
- Cheng, Y. (1995). **_Mean shift, mode seeking, and clustering_**. IEEE transactions on pattern analysis and machine intelligence, 17(8), 790-799.


### Contact
Linh Ma (linh.mavan@gm.gist.ac.kr), Machine Learning & Vision Laboratory, GIST, South Korea