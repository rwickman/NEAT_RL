from neat_rl.args import get_parser
from neat_rl.environments.pendulum import Pendulum
from neat_rl.environments.env import Environment

def main(args):
    env = Environment(args)
    env.train()
    

if __name__ == "__main__":
    args = get_parser().parse_args()

    main(args)