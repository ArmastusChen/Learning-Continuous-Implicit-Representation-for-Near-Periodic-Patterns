# Learning Continuous Implicit Representation for Near-Periodic Patterns


## Get started
You can set up the environment with all dependencies like so:
```
conda create --name NPP-Net python=3.8.5
conda activate NPP-Net
pip install -r requirements.txt
```

## High-Level structure
* data: input examples for completion, remapping, and segmentation.
* externel_lib: externel library to support our code.
* NPP_completion: implementation for completion task. 
* NPP_remapping: implementation for remapping task. 
* NPP_segmentation: implementation for segmentation task. 
* periodicity_searching: implementation for top-K periodicity searching.

## How to Run

1. Please go to this repository (https://github.com/42x00/p3i) and download the pre-trained AlexNet weight in the "Pre-trained Models" section. 

2. Put the downloaded file (alexnet-owt-4df8aa71.pth) in the root of this directory. 




### NPP Completion

Run all examples in the "data/completion/input" using the following command.

```
bash run_completion.sh
```

This script first searches the periodicity of the image, saved in "data/completion/detected". 
Then it performs image completion, generating the outputs in "results/completion_top3". 

The good results for each example can be achieved in 2400 epochs (testset_002400).


