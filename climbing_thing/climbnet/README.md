# Climbnet

## Overview
A wrapper around CLIMBNET from [this repo](https://github.com/juangallostra/climbnet)
## Overview
"Climbnet is a CNN that detects holds on climbing gym walls and returns the appropriate boundary mask for use in instance segmentation."
It utilizes detectron2's Mask-RCNN implementation.

## Installation
A venv is recommended.

I used anaconda to create a venv w/ python 3.8
```
conda create -n <venv name> python=3.8.12
```

You'll need pytorch and torchvision next. The pip requirements file will install detectron2 but complain that torch is missing if you don't do so.

If you wish to use a GPU you'll have to install an appropriate cuda version alongside pytorch<br>
See [the pytorch get started page](https://pytorch.org/get-started/locally/) for what commands to run for what torch and cuda versions. The command will also install torchaudio but it is unecessary.
```
# Example for cuda 11.1 without torchaudio
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Finally install the requirements package in the repo root
```
pip install -r requirements.txt
```

### Running ClimbNet
Download the weights from [this google drive](https://drive.google.com/drive/folders/1MMd7vu9b6XbNrVTxLZ_uehNue5ZBPgnL) and place them in `climbnet/weights/`

Run `Climbing-Thing/climbing_thing/run_climbnet.py`
```
# Uses default test image
python run_climbnet.py
```
or 
```
python run_climbnet.py --image_path <path to your image>
```





