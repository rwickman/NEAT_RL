import wandb
import random, os
from neat_rl.environments.env_pop_diversity_sac import EnvironmentGADiversitySAC
from neat_rl.environments.env_pop_diversity import EnvironmentGADiversity
class HyperparameterTuner:
    def __init__(self, args):
        self.args = args
        self.project = self.args.env
        if self.args.use_td3_diversity:
            self.project = self.project + "-td3"
        

    def start(self):
        sweep_id = self.create_sweep_config()
        wandb.agent(sweep_id, function=self.hyperparameter_tune_main, count=32)

    def hyperparameter_tune_main(self):
        wandb.init(project=self.project, entity="neat")

        self.args.org_lr = wandb.config.org_lr
        self.args.max_species = wandb.config.max_species
        self.args.init_species = wandb.config.max_species
        self.args.init_pop_size = wandb.config.init_pop_size
        self.args.n_org_updates = int(wandb.config.n_org_updates)
        self.args.pg_rate = wandb.config.pg_rate
        self.args.disc_lam = wandb.config.disc_lam

        if self.args.use_td3_diversity:
            self.args.expl_noise = wandb.config.expl_noise
            env = EnvironmentGADiversity(self.args)
        else:
            self.args.sac_alpha = wandb.config.sac_alpha
            env = EnvironmentGADiversitySAC(self.args)

        for epoch in range(self.args.num_episodes):
            
            max_total_reward = env.train()
            wandb.log({
                "epoch": epoch,
                "max_total_reward": max_total_reward,
            })


    def create_sweep_config(self):
        if self.args.use_td3_diversity:

            sweep_configuration = {
                'method': 'bayes',
                'name': 'sweep',
                'metric': {'goal': 'maximize', 'name': 'max_total_reward'},
                'parameters': 
                {
                    'org_lr': {'max': 1e-3, 'min': 1e-5},
                    'max_species': {'values': [1, 2, 3, 4, 5, 6, 7, 8]},
                    'init_pop_size': {'max': 64, 'min': 8},
                    'n_org_updates': {'max': 64, 'min': 8},
                    'pg_rate': {'max': 1.0, 'min': 0.0},
                    'disc_lam': {'max': 1.0, 'min': 0.0},
                    'expl_noise': {'max': 0.5, 'min': 0.0}
                }
            }
        else:
                sweep_configuration = {
                'method': 'bayes',
                'name': 'sweep',
                'metric': {'goal': 'maximize', 'name': 'max_total_reward'},
                'parameters': 
                {
                    'org_lr': {'max': 1e-3, 'min': 1e-5},
                    'max_species': {'values': [1, 2, 3, 4, 5, 6, 7, 8]},
                    'init_pop_size': {'max': 64, 'min': 8},
                    'n_org_updates': {'max': 64, 'min': 8},
                    'pg_rate': {'max': 1.0, 'min': 0.0},
                    'disc_lam': {'max': 1.0, 'min': 0.0},
                    'sac_alpha': {'max': 0.1, 'min': 0.0},

                }
            }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=self.project)

        return sweep_id
