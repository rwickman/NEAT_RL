# NEAT_RL
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