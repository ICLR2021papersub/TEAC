# TEAC

Trust-Entropy Actor-Critic (TEAC) is an off-policy actor-critic method for continuous control. 

Paper is underreviewed, link:[Openreview](https://openreview.net/forum?id=cbtV7xGO9pS)

Combine the trust region and maximum entropy method, TEAC achieves comparable performance on several Mujoco tasks.

The project is based on [OpenAI spinningup](https://github.com/openai/spinningup)

## Requirements

gym[atari,box2d,classic_control]~=0.15.3

tensorflow>=1.8.0,<2.0

torch==1.3.1

mujoco-py==2.0.2.13

## How to run

First, create a conda env and install all dependencies ( this project needs a mujoco license, see [mujoco_website](https://www.roboti.us/license.html) )
  
```
cd /path/to/this/project/

conda create -n teac python=3.6

pip install -e .

pip install mujoco-py
```

Then, we can run the code

`
python teac.py --env Humanoid-v3
`

## Results

### Humanoid-v3
![](https://github.com/ICLR2021papersub/TEAC/blob/master/figures/Humanoid-v3.jpeg "Humanoid-v3")

### Ant-v3
![Ant-v3](https://github.com/ICLR2021papersub/TEAC/blob/master/figures/Ant-v3.jpeg)

### Swimmer-v3
![Swimmer-v3](https://github.com/ICLR2021papersub/TEAC/blob/master/figures/Swimmer-v3.jpeg)

### Walker2d-v3
![Walker2d-v3](https://github.com/ICLR2021papersub/TEAC/blob/master/figures/Walker2d-v3.jpeg)

### Hopper-v3
![Hopper-v3](https://github.com/ICLR2021papersub/TEAC/blob/master/figures/Hopper-v3.jpeg)

### HalfCheetah-v3
![HalfCheetah-v3](https://github.com/ICLR2021papersub/TEAC/blob/master/figures/HalfCheetah-v3.jpeg)

## License

MIT license

## Citing Trust Entropy Actor Critic

If you reference or use TEAC in your research, please cite:
```
@inproceedings{
anonymous2021teac,
title={{\{}TEAC{\}}: Intergrating Trust Region and Max Entropy Actor Critic for Continuous Control},
author={Anonymous},
booktitle={Submitted to International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=cbtV7xGO9pS},
note={under review}
}
```
