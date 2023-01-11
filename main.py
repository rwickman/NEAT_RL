from neat_rl.args import update_parser
from neat.args import get_parser
from neat_rl.environments.env import Environment
from neat_rl.environments.env_pop import EnvironmentGA
from neat_rl.wandb_sweep import HyperparameterTuner
from neat_rl.helpers.saving import save_population
def main(args):
    #env = Environment(args)
    if args.hyperparameter_tune:
        tuner = HyperparameterTuner(args)
        tuner.start()
    else:
        # env = Environment(args)
        # env.train()
        env = EnvironmentGA(args)
        for i in range(args.num_episodes):
            if i % 4 == 0:
                env.td3ga.save()
                save_population(env.population, args.save_file)

            max_fitness = env.train()
            print(f"BEST REWARD IS {max_fitness} FOR EPISODE {i}")
        

if __name__ == "__main__":
    args = update_parser(get_parser()).parse_args()


    main(args)