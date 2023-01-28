import wandb
import random, os
from neat_rl.environments.env_pop_diversity import EnvironmentGADiversity
from neat_rl.helpers.util import load_kdtree
class HyperparameterTuner:
    def __init__(self, args):
        self.args = args
        self.project = self.args.env + "-td3"
        if not self.args.use_state_disc:
            self.project += "-behavior" 
        

    def start(self):
        sweep_id = self.create_sweep_config()
        wandb.agent(sweep_id, function=self.hyperparameter_tune_main, count=32)

    def hyperparameter_tune_main(self):
        wandb.init(project=self.project, entity="neat")

        self.args.num_species = wandb.config.num_species
        self.args.pop_size = wandb.config.pop_size
        self.args.disc_lam = wandb.config.disc_lam
        self.args.survival_rate = wandb.config.survival_rate
        self.args.pg_rate = wandb.config.pg_rate
        
        kdt = load_kdtree(self.args.env)
        archive = {}
        env = EnvironmentGADiversity(self.args, archive, kdt)
        
        epoch = 0
        cur_max_total_reward = None
        while env.total_eval < self.args.max_org_evals:
            max_total_reward, _, _, _ = env.train()
            if len(env.archive) > 0:
                total_fitness_archive = sum(list(archive.values()))
            else:
                total_fitness_archive = 0
            
            if cur_max_total_reward is None or max_total_reward > cur_max_total_reward:
                cur_max_total_reward = max_total_reward

            wandb.log({
                "epoch": epoch,
                "max_total_reward": cur_max_total_reward,
                "coverage": len(env.archive),
                "qd-score": total_fitness_archive
            })
            if env.td3ga.replay_buffer.size >= self.args.batch_size * 4:
                env.population.evolve()
            epoch += 1


    def create_sweep_config(self):
        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'metric': {'goal': 'maximize', 'name': 'max_total_reward'},
            'parameters': 
            {
                'num_species': {'values': [4, 8, 16, 32]},
                'pop_size': {'values': [64, 128, 256]},
                'pg_rate': {'values': [0.5, 0.75, 1.0]},
                'survival_rate': {'max': 0.5, 'min': 0.2},
                'disc_lam': {'max': 0.1, 'min': 0.0}
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=self.project)

        return sweep_id
