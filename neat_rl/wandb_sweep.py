import wandb
import random, os
from neat_rl.environments.env_pop_diversity_sac import EnvironmentGADiversitySAC
from neat_rl.environments.env_pop_diversity import EnvironmentGADiversity
from neat_rl.helpers.util import load_kdtree
class HyperparameterTuner:
    def __init__(self, args):
        self.args = args
        self.project = self.args.env + "-td3"
        

    def start(self):
        sweep_id = self.create_sweep_config()
        wandb.agent(sweep_id, function=self.hyperparameter_tune_main, count=32)

    def hyperparameter_tune_main(self):
        wandb.init(project=self.project, entity="neat")

        self.args.num_species = wandb.config.num_species
        self.args.pop_size = wandb.config.pop_size
        self.args.n_org_updates = int(wandb.config.n_org_updates)
        self.args.disc_lam = wandb.config.disc_lam
        self.args.survival_rate = wandb.config.survival_rate

        
        kdt = load_kdtree(self.args.env)
        archive = {}
        env = EnvironmentGADiversity(self.args, archive, kdt)
        
        epoch = 0
        while env.total_eval < self.args.max_org_evals:       
            print("env.total_eval ", env.total_eval )
            max_total_reward, _, _, _ = env.train()
            if len(env.archive) > 0:
                total_fitness_archive = sum(list(archive.values()))
            else:
                total_fitness_archive = 0

            wandb.log({
                "epoch": epoch,
                "max_total_reward": max_total_reward,
                "coverage": len(env.archive),
                "qd-score": total_fitness_archive
            })
            if env.td3ga.replay_buffer.size >= self.args.batch_size * 4:
                env.population.evolve()
            epoch += 1


    def create_sweep_config(self):
            # sweep_configuration = {
            #     'method': 'random',
            #     'name': 'sweep',
            #     'metric': {'goal': 'maximize', 'name': 'max_total_reward'},
            #     'parameters': 
            #     {
            #         'org_lr': {'max': 1e-3, 'min': 1e-5},
            #         'num_species': {'values': [1, 2, 3, 4, 5, 6, 7, 8]},
            #         'pop_size': {'max': 64, 'min': 16},
            #         'survival_rate': {'max': 0.8, 'min': 0.1},
            #         'n_org_updates': {'max': 64, 'min': 8},
            #         'pg_rate': {'max': 1.0, 'min': 0.0},
            #         'disc_lam': {'max': 1.0, 'min': 0.0},
            #         'expl_noise': {'max': 0.5, 'min': 0.0},
            #         'update_freq': {'max': 64, 'min': 16}
            #     }
            # }

        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'metric': {'goal': 'maximize', 'name': 'max_total_reward'},
            'parameters': 
            {
                'num_species': {'values': [4, 8, 16, 32]},
                'pop_size': {'values': [64, 128, 256]},
                'survival_rate': {'max': 0.6, 'min': 0.2},
                'n_org_updates': {'values': [32, 64, 128]},
                'disc_lam': {'max': 2.0, 'min': 0.0}
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=self.project)

        return sweep_id
