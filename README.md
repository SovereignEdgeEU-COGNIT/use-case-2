# Use Case 2 - Wildfire Detection
This repository contains the wildfire image recognition function offloaded to the COGNIT framework to perform the preliminary tests and the basic FaaS call for COGNIT V2, some images to tests the algorithm on as well as the statistical demo for the simulation of a wildfire.

The repository contains an `examples` folder where the software and the images to run the tests are stored (more information about the COGNIT device runtime can be found at: https://github.com/SovereignEdgeEU-COGNIT/device-runtime-py) and a `FireUC` folder that should be uploaded to the COGNIT framework image to set up the image itself. It contains a `requirements.txt` file to install the required Python libraries as well as the *pre-trained neural network model* in TensorFlow format imported in the function.

## Neural network model
The pre-trained neural network model used for this test can be found at the following link:
https://github.com/tobybreckon/fire-detection-cnn. Minor changes have been performed to the provided software to adapt it to our needs.

## Setting up the image
The virtual machine requirements to run the image recognition function are a memory size of at least 3.072 GB and a disk size of 8 GB. Additionally, it must support SSE4.1 instructions to set the VM template CPU mode on "host-passthrough".

To set up the image, upload the FireUC folder to the *VM* in `/root/FireUC`. 
Then, activate the virtual environment (https://github.com/SovereignEdgeEU-COGNIT/serverless-runtime):
```
source /root/serverless-runtime/serverless-env/bin/activate
```
`libGL` library (required for the installation of `opencv-contrib-python`) can be installed on openSUSE distribution through the following commands:
```
zypper install Mesa-libGL1
zypper install libgthread-2_0-0
```
Finally, the requirements listed into `/root/FireUC/requirements.txt` can be installed:
```
pip install -r requirements.txt
```
## Function offload
To run `FireUC_offload.py`, the variable REQ_TIMEOUT variable in the Serverless Runtime Client (https://github.com/SovereignEdgeEU-COGNIT/device-runtime-py/blob/main/cognit/modules/_serverless_runtime_client.py) should be increased to 20 to give TensorFlow enough time to be loaded. The image to analyse is hardcoded into the software itself into the "image" variable and must be in *png* format. The function returns:
* 0: a fire is NOT detected
* 1: a fire is detected
* 2: an error occurred during the execution of the function

## Statistical model
The statistical model (`uc2_statistical_model.py`) shows a fire starting from the center and spreading out. When running the program a Matplotlib window will open showing the fire spreading (yellow is fire, purple is no fire). The image needs to wait for all sensors to "respond" then it will refresh, it may require some time.
