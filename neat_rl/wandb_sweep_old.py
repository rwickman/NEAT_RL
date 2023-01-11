import wandb
import random, os
from neat_rl.environments.env_pop_sweep import EnvironmentGA

class HyperparameterTuner:
    def __init__(self, args):
        self.args = args
        self.project = self.args.env

    def start(self):
        sweep_id = self.create_sweep_config()
        wandb.agent(sweep_id, function=self.hyperparameter_tune_main, count=32)

    def hyperparameter_tune_main(self):
        wandb.init(project=self.project, entity="neat")
        
        #self.args.save_dir = f"{self.models_path}/models_{len(os.listdir(self.models_path))}"

        self.args.learning_starts = wandb.config.learning_starts
        self.args.lr = wandb.config.lr
        self.args.actor_lr = wandb.config.actor_lr
        self.args.org_lr = wandb.config.org_lr
        self.args.max_species = wandb.config.max_species
        self.args.init_species = wandb.config.max_species
        self.args.init_pop_size = wandb.config.init_pop_size
        self.args.n_org_updates = int(wandb.config.n_org_updates)
        self.args.batch_size = wandb.config.batch_size
        self.args.pg_rate = wandb.config.pg_rate

        env = EnvironmentGA(self.args)

        for epoch in range(self.args.num_episodes):
            max_total_reward = env.train()
            wandb.log({
                "epoch": epoch,
                "max_total_reward": max_total_reward,
            })
            if max_total_reward == 501:
                break

    def create_sweep_config(self):
        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'metric': {'goal': 'maximize', 'name': 'max_total_reward'},
            'parameters': 
            {
                'learning_starts': {'max': 2000, 'min': 500},
                'actor_lr': {'max': 1e-3, 'min': 1e-5},
                'lr': {'max': 1e-3, 'min': 1e-5},
                'org_lr': {'max': 1e-3, 'min': 1e-5},
                'max_species': {'values': [1, 2, 3, 4, 5, 6, 7, 8]},
                'init_pop_size': {'values': [8, 16, 32, 64]},
                'n_org_updates': {'max': 64, 'min': 8},
                'batch_size': {'values': [64, 128, 256]},
                'pg_rate': {'max': 1.0, 'min': 0.0},
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=self.project)

        return sweep_id
