# NEAT_RL
## Hyperparmeter tunning
To hyperparameter tune on the QDHopper environment: 
```shell
python3 main.py --hyperparameter_tune --env QDHopperBulletEnv-v0 --max_org_evals 10000 --use_state_disc`
```

For other environments, replace the the environment with the one you want to test on.

All environments:
* QDHopperBulletEnv-v0
* QDWalker2DBulletEnv-v0
* QDHalfCheetahBulletEnv-v0
* QDAntBulletEnv-v0