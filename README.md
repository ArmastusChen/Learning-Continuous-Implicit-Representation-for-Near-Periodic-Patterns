# Learning Continuous Implicit Representation for Near-Periodic Patterns (ECCV 2022)


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


### NPP Completion

Run all examples in the "data/completion/input" using the following command.

```
bash run_completion.sh
```

This script performs image completion, generating the outputs in "results/completion_top3". 

The good results for each example can be achieved in 2400 epochs (testset_002400).


