# SRPO
[NeurIPS 2023] The official code for paper ["State Regularized Policy Optimization on Data with Dynamics Shift"](https://arxiv.org/abs/2306.03552).


## Installation

Follow the steps in [OfflineRL](https://github.com/polixir/OfflineRL)

## Prepare Offline Dataset

Download the files in [Google Drive](https://drive.google.com/file/d/19Bc8LSE38A67LH3ZCaZDXDuHEc7tC35G/view?usp=sharing) and change the `path` parameter in line:15 of `examples/train_d4rl.py`.


## Run the SRPO algorithm

```
python examples/train_d4rl.py --algo_name=maple_st --exp_name=maple_st --seed 1 --task density_10,body_mass@walker2d-medium-expert-v0 --rew_reg_eta 0.1 --out_train_epoch 200 --device cuda:1
```

`walker2d-medium-expert-v0` can be changed to other Offline RL environments. To run baseline algorithms, `maple_st` can be changed to `maple`, `mopo`, `cql`, etc.

## Citation

If you find our code repository or paper useful, please cite with:

@article{xue2023state,
  title={State Regularized Policy Optimization on Data with Dynamics Shift},
  author={Xue, Zhenghai and Cai, Qingpeng and Liu, Shuchang and Zheng, Dong and Jiang, Peng and Gai, Kun and An, Bo},
  journal={arXiv preprint arXiv:2306.03552},
  year={2023}
}
