# libtorch_2022_demo
# PyTorch and cuDNN Setup for Visual Studio 2022

This guide provides steps to set up PyTorch and cuDNN for a C++ project in Visual Studio 2022.

## Prerequisites

- [Visual Studio 2022](https://visualstudio.microsoft.com/vs/)

- [vcpkg](https://github.com/microsoft/vcpkg#quick-start-windows)

## Installation

1. Download [PyTorch](https://pytorch.org/get-started/locally/). Follow the instructions on the PyTorch website to install it on your system.

2. Download [cuDNN 11.8](https://developer.nvidia.com/cudnn). Make sure you select a version that is compatible with the version of PyTorch you've installed. Follow NVIDIA's instructions to install it on your system.

3. Install dependencies using vcpkg in Visual Studio 2022. The following libraries are required for the project:

    - torch
    - opencv

    You can install them using the following commands:

    ```shell
    vcpkg install libtorch:x64-windows
    vcpkg install opencv4:x64-windows
    ```

    Include the necessary headers in your source files:

    ```cpp
    #include <torch/script.h>
    #include <torch/torch.h>
    #include <iostream>
    #include <memory>
    // opencv
    #include <opencv2/opencv.hpp>
    #include <opencv2/core.hpp>
    #include <opencv2/imgproc.hpp>
    ```

4. Build the project in Visual Studio. You can do this by selecting `Build -> Build Solution` from the menu or by pressing `Ctrl+Shift+B`.

5. Run the project. Press `Ctrl+F5` to start the project without debugging.

## Troubleshooting

If you encounter any issues while setting up or running the project, feel free to raise an issue in this repository.

