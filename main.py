import os

from neat_rl.args import update_parser
from neat.args import get_parser
from neat_rl.environments.env import Environment
from neat_rl.environments.env_pop import EnvironmentGA
from neat_rl.environments.env_sac import EnvironmentSAC
from neat_rl.environments.env_pop_diversity import EnvironmentGADiversity
from neat_rl.environments.env_pop_diversity_sac import EnvironmentGADiversitySAC
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
        #env = EnvironmentGA(args)
        if args.use_td3:
            env = Environment(args)
            env.train()
        elif args.use_sac:
            env = EnvironmentSAC(args)
            env.train()
        else:
            if args.use_td3_diversity:
                env = EnvironmentGADiversity(args)
                for i in range(args.num_episodes):        
                    max_fitness = env.train()
                    if not args.render:
                        env.td3ga.save()
                        save_population(env.population, args.save_file)

                    print(f"BEST REWARD IS {max_fitness} FOR EPISODE {i}")
            else:    
                env = EnvironmentGADiversitySAC(args)
                for i in range(args.num_episodes):            
                    max_fitness = env.train()
                    if not args.render:
                        env.sac.save()
                        save_population(env.population, args.save_file)

                    print(f"BEST REWARD IS {max_fitness} FOR EPISODE {i}")
                

if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = update_parser(get_parser())
    parser.add_argument("--render", action="store_true",
        help="Render frames.")
    parser.add_argument("--use_td3", action="store_true",
        help="Use normal TD3.")
    parser.add_argument("--use_sac", action="store_true",
        help="Use normal SAC.")
    parser.add_argument("--use_td3_diversity", action="store_true",
        help="Use normal SAC.")

    main(parser.parse_args())