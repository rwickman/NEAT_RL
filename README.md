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
