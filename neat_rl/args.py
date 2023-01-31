import argparse

def update_parser(parser):
    parser.add_argument("--env", default="QDHalfCheetahBulletEnv-v0",
        help="Environment to test (e.g., QDHopperBulletEnv-v0, QDWalker2DBulletEnv-v0, QDHalfCheetahBulletEnv-v0, etc.).")
    parser.add_argument("--load", action="store_true",
        help="Load the models.")

    parser.add_argument("--lr", type=float, default=3e-4,
        help="Learning rate for actor and critic networks.")
    parser.add_argument("--actor_lr", type=float, default=3e-4,
        help="Learning rate for actor and critic networks.")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="Gamma used for learning rate discount.")
    parser.add_argument("--tau", type=float, default=0.005,
        help="Value used for interpolating target policy.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="Batch size for updating networks.")
    parser.add_argument("--replay_capacity", type=int, default=524288,
        help="Maximum size of replay memory.")
    parser.add_argument("--behavior_capacity", type=int, default=262144,
        help="Maximum size of replay memory.")
    parser.add_argument("--max_norm", type=float, default=2.0,
        help="Maximum norm of gradient update.")

    parser.add_argument("--n_hidden", type=int, default=1,
        help="Number of hidden layers.")
    parser.add_argument("--hidden_size", type=int, default=128,
        help="Hidden size of network.")
    parser.add_argument("--critic_hidden_size", type=int, default=256,
        help="Hidden size of critic network.")

    parser.add_argument("--expl_noise", type=float, default=0.2,
        help="Exploration noise.")
    
    parser.add_argument("--policy_noise", type=float, default=0.2,
        help="Noise used in policy prediction.")
    parser.add_argument("--noise_clip", type=float, default=0.5,
        help="Noise used in policy prediction.")
    parser.add_argument("--policy_freq", type=int, default=2,
        help="How often to update the policy w.r.t. the critic.")
    parser.add_argument("--update_freq", type=int, default=8,
        help="How often to update the policy and critic.")


    parser.add_argument("--learning_starts", type=int, default=1000,
        help="Number of timesteps to elapse before training.")


    parser.add_argument("--save_dir", default="models/",
        help="Directory to save the modelsa and results.")

    # parser.add_argument("--env", default="InvertedPendulum-v4",
    #     help="Environment to run (e.g., InvertedPendulum-v4, Pendulum-v1).")
    # parser.add_argument("--max_timesteps", type=int, default=50000,
    #     help="Hidden size of network.")


    parser.add_argument("--org_lr", type=float, default=6e-4,
        help="Learning rate for an organism using policy updates.")
    parser.add_argument("--n_org_updates", type=int, default=64,
        help="Number of updates to perform for an organism.")
    parser.add_argument("--pg_rate", type=float, default=1.0,
        help="Probability of performing policy gradient updates instead of GA updates.")
    parser.add_argument("--hyperparameter_tune", action="store_true",
        help="Tune the hyperparameters.")
    parser.add_argument("--num_episodes", type=int, default=50,
        help="Number of episodes to run.")

    parser.add_argument("--disc_lr", type=float, default=3e-4,
        help="Learning rate of species discriminator.")
    parser.add_argument("--disc_lam", type=float, default=0.05,
        help="Reward scaling for discriminator.")
    parser.add_argument("--disc_train_iter", type=int, default=64,
        help="Number of iterations to train the discriminator.")

    parser.add_argument("--use_state_disc", action="store_true",
        help="Use state/actions discriminator.")  
    parser.add_argument("--use_state_only_disc", action="store_true",
        help="Use state/actions discriminator.")        
    parser.add_argument("--use_action_disc", action="store_true",
        help="Use action discriminator.") 
    parser.add_argument("--no_train_diversity", action="store_true",
        help="Don't train the critic with diversity bonus, only for selecting organsims in a species.")        

    parser.add_argument("--no_use_disc", action="store_true",
        help="Don't use the discriminator to increase diversity.")
    parser.add_argument("--sac_alpha", type=float, default=0.2,
        help="Alpha used for entropy in SAC.")

    parser.add_argument("--iso_sigma", type=float, default=0.005,
        help="ISO sigma for random noise.")
    parser.add_argument("--line_sigma", type=float, default=0.05,
        help="Line sigma for interpolation noise.")
    parser.add_argument("--skew_val", type=float, default=-1.0,
        help="Value used for skewing behavior distribution.")


    parser.add_argument("--num_species", type=int, default=8,
        help="Number of species to create.")
    parser.add_argument("--pop_size", type=int, default=64,
        help="Population size.")
    parser.add_argument("--max_org_evals", type=int, default=1e5, 
        help="Total number of organism evaluations to run.")
    parser.add_argument("--survival_rate", type=float, default=0.5, 
        help="Percentage of organisms that will survive.")
    parser.add_argument("--diversity_bonus_sort", action="store_true",
        help="Sort by organisms by diversity bonus reward.")
    parser.add_argument("--best_diversity_sort", action="store_true",
        help="Sort by organisms by best fitness score.")
    parser.add_argument("--random_sort", action="store_true",
        help="Sort organisms randomly.")
    parser.add_argument("--only_pg", action="store_true",
        help="Only perform PG updates.")
    parser.add_argument("--resample_species", action="store_true",
        help="Resample the species for trajectories.")    
    parser.add_argument("--resample_behavior", action="store_true",
        help="Resample the species for trajectories.")
    parser.add_argument("--resample_behavior_prob", type=float, default=0.1,
        help="Probability of resampling a behavior.")
   
    return parser
    
