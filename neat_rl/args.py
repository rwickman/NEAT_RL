import argparse

def get_parser():
    parser = argparse.ArgumentParser()
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


    parser.add_argument("--save_file", default="models/td3.pt",
        help="Location to save the models.")

    parser.add_argument("--env", default="InvertedPendulum-v4",
        help="Environment to run (e.g., InvertedPendulum-v4, Pendulum-v1).")
    parser.add_argument("--max_timesteps", type=int, default=500,
        help="Hidden size of network.")
    parser.add_argument("--load", action="store_true",
        help="Load the models.")


    return parser
    
