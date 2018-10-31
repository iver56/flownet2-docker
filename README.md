# FlowNet 2.0 Docker Image

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

This repository contains a Dockerfile and scripts to build and run neural networks for optical flow estimation in Docker containers. We also provide some example data to test the networks. 

![Teaser](data/teaser.png)

If you use this project or parts of it in your research, please cite the original paper of Flownet 2.0:


    @InProceedings{flownet2,
      author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
      title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = "Jul",
      year         = "2017",
      url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
    }


See the [paper website](https://lmb.informatik.uni-freiburg.de/Publications/2017/IMKDB17) for more details.


## 0. Requirements

We use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quick-start) for reliable GPU support in the containers. This is an extension to Docker and can be easily installed with just two commands.

To run the FlowNet2 networks, you need an Nvidia GPU (at least Kepler). For the smaller networks (e.g. *FlowNet2-s*) 1GB of VRAM is sufficient, while for the largest networks (the full *FlowNet2*) at least **4GB** must be available. A GTX 970 can handle all networks.

## 1. Building the FN2 Docker image

Simply run `make`. This will create two Docker images: The OS base (an Ubuntu 16.04 base extended by Nvidia, with CUDA 8.0), and the "flownet2" image on top. In total, about 8.5GB of space will be needed after building. Build times are a little slow.


## 2. Running an FN2 container as a server

To start a web server that serves flownet2 as a REST API, run

`docker run --rm --runtime nvidia --network host -it flownet2`

Or in background mode:

`docker run -d --runtime nvidia --network host flownet2`

### API documentation

`POST /estimate_flow/` with data payload like this:

```json
{
  "image1_base64": "First base64 encoded PNG image goes here",
  "image2_base64": "Second base64 encoded PNG image goes here"
}
```

The output matrix will have 512 x 512 resolution, regardless of input size.
This matrix needs to be rescaled (e.g. with
[scikit-image](http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize) with `preserve_range=True`)
to the input image size if needed.

The returned JSON will look like this:

```
{
  "flow": [[[-0.44253078  0.13436875]
            [-0.5787499   0.42568296]
            [-0.59728986  0.42983937]
            ...
            [ 1.52206624 -0.2325332 ]
            [ 1.28677487 -0.21145242]
            [ 0.799366   -0.39404145]]
          
           [[-0.52459794  0.42612121]
            [-0.60560703  0.3781198 ]
            [-0.55513763  0.43691179]
            ...
            [ 1.49052036 -0.13133171]
            [ 1.46423972 -0.24987227]
            [ 1.4011569  -0.30543262]]
          
           [[-0.55959642  0.40541875]
            [-0.61500663  0.45374426]
            [-0.52857304  0.41693285]
            ...
            [ 1.56535566 -0.21450724]
            [ 1.50720096 -0.19533606]
            [ 1.46007121 -0.12436457]]
          
           ...
          
           [[-1.00231111  0.03182566]
            [-1.00792384  0.03718051]
            [-1.03274596 -0.01514795]
            ...
            [-0.05430229  1.16414952]
            [ 0.02033724  1.23582554]
            [ 0.44144398  0.94685918]]
          
           [[-1.00950956  0.00764451]
            [-1.18153584  0.00676027]
            [-1.13824499  0.008102  ]
            ...
            [-0.15946357  1.11376107]
            [ 0.22915983  1.25269079]
            [ 0.13352813  1.42669117]]
          
           [[-1.00627697  0.1358473 ]
            [-1.12843215 -0.08414076]
            [-1.21641994  0.00675413]
            ...
            [ 0.03442889  1.34803653]
            [-0.23310344  0.87845743]
            [ 0.22394601  0.79704392]]],
  "scale_x_back_factor": 4.0,
  "scale_y_back_factor": 4.0
}
```


## 3. License
The files in this repository are under the [GNU General Public License v3.0](LICENSE)

