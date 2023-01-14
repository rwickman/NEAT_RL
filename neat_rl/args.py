import argparse

def update_parser(parser):
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
    parser.add_argument("--replay_capacity", type=int, default=131072,
        help="Maximum size of replay memory.")

    parser.add_argument("--n_hidden", type=int, default=1,
        help="Number of hidden layers.")
    parser.add_argument("--hidden_size", type=int, default=32,
        help="Hidden size of network.")
    parser.add_argument("--expl_noise", type=float, default=0.1,
        help="Exploration noise.")
    
    parser.add_argument("--policy_noise", type=float, default=0.2,
        help="Noise used in policy prediction.")
    parser.add_argument("--noise_clip", type=float, default=0.5,
        help="Noise used in policy prediction.")
    parser.add_argument("--policy_freq", type=int, default=2,
        help="How often to update the policy w.r.t. the critic.")

    parser.add_argument("--learning_starts", type=int, default=10000,
        help="Number of timesteps to elapse before training.")


    parser.add_argument("--rl_save_file", default="models/td3.pt",
        help="Location to save the models.")

    # parser.add_argument("--env", default="InvertedPendulum-v4",
    #     help="Environment to run (e.g., InvertedPendulum-v4, Pendulum-v1).")
    parser.add_argument("--max_timesteps", type=int, default=50000,
        help="Hidden size of network.")


    parser.add_argument("--org_lr", type=float, default=3e-4,
        help="Learning rate for an organism using policy updates.")
    parser.add_argument("--n_org_updates", type=int, default=32,
        help="Number of updates to perform for an organism.")
    parser.add_argument("--pg_rate", type=float, default=0.5,
        help="Probability of performing policy gradient updates instead of GA updates.")        
    parser.add_argument("--emb_size", type=int, default=128,
        help="Embedding size for species.")
    parser.add_argument("--hyperparameter_tune", action="store_true",
        help="Tune the hyperparameters.")
    parser.add_argument("--num_episodes", type=int, default=50,
        help="Number of episodes to run.")

    parser.add_argument("--disc_lr", type=float, default=3e-4,
        help="Probability of performing policy gradient updates instead of GA updates.")
    parser.add_argument("--disc_lam", type=float, default=0.1,
        help="Reward scaling for discriminator.")
        
    parser.add_argument("--no_use_disc", action="store_true",
        help="Don't use the discriminator to increase diversity.")

    parser.add_argument("--sac_alpha", type=float, default=0.2,
        help="Alpha used for entropy in SAC.")

    return parser
    
