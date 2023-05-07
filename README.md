# Sam's Backyard Bird Mic

Adapted from https://github.com/kahst/BirdNET

### Install tflite from wheel

```
$ sudo apt install swig libjpeg-dev zlib1g-dev python3-dev \
                   unzip wget python3-pip curl git cmake make
$ sudo pip3 install numpy==1.22.1
$ wget "https://raw.githubusercontent.com/PINTO0309/TensorflowLite-bin/main/2.8.0/download_tflite_runtime-2.8.0-cp39-none-linux_aarch64.whl.sh"
$ chmod +x download_tflite_runtime-2.8.0-cp39-none-linux_aarch64.whl.sh
$ ./download_tflite_runtime-2.8.0-cp39-none-linux_aarch64.whl.sh
$ sudo pip3 install --upgrade tflite_runtime-2.8.0-cp39-none-linux_aarch64.whl
```
