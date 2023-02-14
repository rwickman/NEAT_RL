import os, json, time
from neat_rl.helpers.util import load_kdtree, add_to_archive, save_archive, load_archive

from neat_rl.args import update_parser
from neat_rl.environments.env import Environment
from neat_rl.environments.env_pop import EnvironmentGA
from neat_rl.environments.env_sac import EnvironmentSAC
from neat_rl.environments.env_pop_diversity import EnvironmentGADiversity
from neat_rl.environments.env_pop_diversity_org import EnvironmentGADiversityOrg
from neat_rl.environments.env_pop_diversity_sac import EnvironmentGADiversitySAC


from neat_rl.helpers.saving import save_population

def main(args):
    #env = Environment(args)
    if args.hyperparameter_tune:
        from neat_rl.wandb_sweep import HyperparameterTuner
        tuner = HyperparameterTuner(args)
        tuner.start()
    else:
        if args.use_td3:
            env = Environment(args)
            env.train()
        elif args.use_sac:
            env = EnvironmentSAC(args)
            env.train()
        elif args.non_qd:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            env = EnvironmentGADiversityOrg(args)
            i = 0
            while env.total_eval < args.max_org_evals:     
                i += 1
                max_fitness, avg_fitness, fitness_range, total_fitness = env.train()
                if not args.render and env.td3ga.replay_buffer.size >= args.learning_starts:
                    env.population.evolve()
                save_population(env.population, args.save_dir)
                env.td3ga.save()
                    
                print(f"BEST REWARD IS {max_fitness} AVG FITNESS IS {avg_fitness} FOR EPISODE {i} EVALS {env.total_eval}")
                   

        else:
     
            kdt = load_kdtree(args.env)
            
            
            train_json_file = os.path.join(args.save_dir, f"train_dict_{args.env}.json")
            archive_file = os.path.join(args.save_dir, f"archive_{args.env}.json")
            args_file = os.path.join(args.save_dir, "args.json")
            if args.load and not args.render:
                if not os.path.exists(archive_file):
                    archive_file = os.path.join(args.save_dir, "archive.json") 
                with open(train_json_file) as f:
                    train_dict = json.load(f)
                
                archive, archive_species_ids = load_archive(archive_file)
                archive_file = os.path.join(args.save_dir, f"archive_{args.env}.json")
            else:
                train_dict = {
                    "max_fitness": [],
                    "total_fitness": [],
                    "total_fitness_archive": [],
                    "fitness_range": [],
                    "avg_fitness": [],
                    "coverage": [],
                    "best_org_avg_fitness": [],
                    "best_org_first_fitness": [],
                    "total_evals": 0
                }
                archive = {}
                archive_species_ids = {}

            # if args.use_td3_diversity:
            env = EnvironmentGADiversity(args, archive, archive_species_ids, kdt)
            # else:
            #     env = EnvironmentGADiversitySAC(args, archive, kdt)

            env.total_eval = train_dict["total_evals"]

            i = 0

            
            if not args.render:
                # Create the save directory if it doesn't exist
                if not os.path.exists(args.save_dir):                        
                    os.makedirs(args.save_dir)
                elif not args.load:
                    import sys
                    
                    print("ERROR DIRECTORY {args.save_dir} ALREADY EXISTS")
                    sys.exit()
                    
                # Save the args
                with open(args_file, 'w') as f:
                    json.dump(args.__dict__, f, indent=2)
                


            while env.total_eval < args.max_org_evals:        
                i += 1
                
                if not args.render:

                    # Run all the organisms
                  
                    max_fitness, avg_fitness, fitness_range, total_fitness = env.train()
                  


                    if len(archive) > 0:
                        total_fitness_archive = sum(list(archive.values()))
                        best_fitness = max(list(archive.values()))
                    else:
                        total_fitness_archive = 0
                        best_fitness = max_fitness
                    if len(train_dict["max_fitness"]) >= 1:
                        cur_max_fitness = max(train_dict["max_fitness"][-1], best_fitness)
                    else:
                        cur_max_fitness = best_fitness


                    best_org = env.population.get_best()
                    if i % 16 == 0:
                        best_org_avg_fitness, best_org_first_fitness, eval_max_fitness = env.evaluate_10(best_org)
                        train_dict["best_org_avg_fitness"].append(best_org_avg_fitness) 
                        train_dict["best_org_first_fitness"].append(best_org_first_fitness)
                        cur_max_fitness = max(cur_max_fitness, eval_max_fitness)

                    train_dict["max_fitness"].append(cur_max_fitness)
                    train_dict["total_fitness"].append(total_fitness)
                    train_dict["total_fitness_archive"].append(total_fitness_archive)
                    train_dict["fitness_range"].append(fitness_range)
                    train_dict["avg_fitness"].append(avg_fitness)
                    train_dict["coverage"].append(len(archive))
                    train_dict["total_evals"] = env.total_eval
                    
                    # if i % 2 == 0:
                    start_time = time.time()
                    env.td3ga.save()            
                    save_population(env.population, args.save_dir)
                    save_archive(archive, archive_species_ids, archive_file)
                    with open(train_json_file, "w") as f:
                        json.dump(train_dict, f)

                    print("SAVING TIME", time.time() - start_time)



                    # Evolve the population if not rendering and a minimum number of trajectories have been collected
                    if not args.render and env.td3ga.replay_buffer.size >= args.learning_starts:
                        start_time = time.time()
                        env.population.evolve()
                        print("EVOLVE TIME: ", time.time() - start_time)


                else:
                    max_fitness, avg_fitness, fitness_range, total_fitness = env.train()
                    
                print(f"BEST REWARD IS {max_fitness} AVG FITNESS IS {avg_fitness} FOR EPISODE {i} EVALS {env.total_eval}")
                   
            # else:    
            #     env = EnvironmentGADiversitySAC(args)
            #     for i in range(args.num_episodes):            
            #         max_fitness = env.train()
            #         if not args.render:
            #             env.sac.save()
            #             save_population(env.population, args.save_file)

                    # print(f"BEST REWARD IS {max_fitness} FOR EPISODE {i}")
                

if __name__ == "__main__":
    #import torch
    #torch.multiprocessing.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser = update_parser(parser)
    parser.add_argument("--render", action="store_true",
        help="Render frames.")
    parser.add_argument("--use_td3", action="store_true",
        help="Use normal TD3.")
    parser.add_argument("--use_sac", action="store_true",
        help="Use normal SAC.")
    parser.add_argument("--use_td3_diversity", action="store_true",
        help="Use normal SAC.")
    parser.add_argument("--non_qd", action="store_true",
        help="Use normal SAC.")
    parser.add_argument("--max_episode_steps", type=int, default=1000,
        help="Max episode steps.")
    main(parser.parse_args())