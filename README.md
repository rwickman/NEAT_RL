# NEAT_RL

Codebase for the [paper](https://arxiv.org/pdf/2304.07425.pdf): Efficient Quality-Diversity Optimization through Diverse Quality Species

## Setup
To install this package run:
```shell
pip3 install -e .
```
You will also need to install QDGym:
```shell
pip3 install git+https://github.com/ollenilsson19/QDgym.git#egg=QDgym
```

## Training the models
To train the model on the QDHopper environment with a population size of 64 with 8 species, survival rate of 0.5, and discriminator lambda of 0.05:
```shell
python3 main.py --env QDHopperBulletEnv-v0 --pop_size 64 --num_species 8 --disc_lam 0.05 --survival_rate 0.5 --use_state_disc --save_dir <path/to/save_dir>
```

To load the model to continue training:
```shell
python3 main.py --env QDHopperBulletEnv-v0 --pop_size 64 --num_species 8 --disc_lam 0.05 --survival_rate 0.5 --use_state_disc --save_dir <path/to/save_dir> --load
```


If you want to see population interacting with environment, without training (i.e., render):
```shell
python3 main.py --env QDHopperBulletEnv-v0 --pop_size 64 --num_species 8 --disc_lam 0.05 --survival_rate 0.5 --use_state_disc --save_dir <path/to/save_dir> --load --render
```

To track the current training results for a model:
```shell
python3 neat_rl/helpers/plot_results.py --save_dir <path/to/model> --env <env_name>
```

## Hyperparmeter tunning
To hyperparameter tune on the QDHopper environment: 
```shell
python3 main.py --hyperparameter_tune --env QDHopperBulletEnv-v0 --max_org_evals 10000 --use_state_disc
```

For other environments, replace the environment with the one you want to test on.

All environments:
* QDHopperBulletEnv-v0
* QDWalker2DBulletEnv-v0
* QDHalfCheetahBulletEnv-v0
* QDAntBulletEnv-v0

----

## Hyperpameters used 
<p align="center">

| Hyperparameter                               | Value       |
|----------------------------------------------|-------------|
| Population Size                              | 64          |
| Number of Species ($m$)                      | 8           |
| Diversity Reward Scale ($\lambda$)           | 0.05        |
| Species Elites Value ($K$)                   | 4           |
| Policy Update Steps (n_grad)                 | 64          |
| Critic Update Frequency (critic_update_freq)     | 8           |
| Policy Hidden Size                           | 128         |
| Species Actor Hidden Size                    | 256         |
| Species Critic Hidden Size                   | 256         |
| Discriminator Hidden Size                    | 256         |
| Species Actor/Critic and Discriminator Learning Rate | 0.003 |
| Policy Learning Rate                         | 0.006       |
| Number of Evaluations (num_eval)             | $10^{5}$    |
| Batch Size ($N$)                             | 256         |
| Discount Factor ($\gamma$)                   | 0.99        |
| Species Target Update Rate ($\tau$)          | 0.005       |
| TD3 Exploration Noise                        | 0.2         |
| TD3 Smoothing Variance ($\sigma$)            | 0.2         |
| TD3 Noise Clip ($c$)                         | 0.5         |
| TD3 Target Update Freq. ($d$)                | 2           |
| Replay Buffer Size                           | $2^{19}$    |

**Table 1:** Hyperparameter values for the DQS algorithm.

</p>

-------
## Cite

```
@article{wickman2023efficient,
  title={Efficient Quality-Diversity Optimization through Diverse Quality Species},
  author={Wickman, Ryan and Poudel, Bibek and Villarreal, Michael and Zhang, Xiaofei and Li, Weizi},
  journal={arXiv preprint arXiv:2304.07425},
  year={2023},
  url={https://arxiv.org/pdf/2304.07425.pdf}
}
```
