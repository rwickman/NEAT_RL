import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt

def print_stats(save_dir):
    
    pop_file = os.path.join(save_dir, "pop.json")    
    train_dict_file = os.path.join(save_dir, "train_dict.json")
    with open(pop_file) as f:
        pop_dict = json.load(f)

    with open(train_dict_file) as f:
        train_dict = json.load(f)

    
    max_age = 0
    max_generation = 0
    avg_generation = 0
    min_generation = 1e6
    avg_age = 0

    for org_dict in pop_dict["orgs"]:
        max_age = max(org_dict["age"], max_age)
        max_generation = max(org_dict["generation"], max_generation)
        min_generation = min(org_dict["generation"], min_generation)
        avg_generation += org_dict["generation"]
        avg_age += org_dict["age"]
    
    print("TOTAL GENERATIONS:", pop_dict["generation"])
    print("MAX AGE:", max_age)
    print("AVG AGE: ", avg_age/len(pop_dict["orgs"]))
    print("MAX GENERATION:", max_generation)
    print("MIN GENERATION:", min_generation)
    print("AVG GENERATION:", avg_generation/len(pop_dict["orgs"]))
    print("COVERAGE LAST 10:", train_dict["coverage"][-10:])
    print("total_fitness_archive", train_dict["total_fitness_archive"][-5:])
    print("MAX FITNESS:", train_dict["max_fitness"][-1])
        
def plot(save_dir):
    train_dict_file = os.path.join(save_dir, "train_dict.json")

    def moving_average(x, w=3):
        return np.convolve(x, np.ones(w), 'valid') / w

    with open(train_dict_file) as f:
        train_dict = json.load(f)
    

    fig, axs = plt.subplots(3)
    
    axs[0].plot(moving_average(train_dict["total_fitness_archive"]))
    axs[0].set(ylabel="QD-Score")

    axs[1].plot(moving_average(train_dict["coverage"]))
    
    axs[1].set(ylabel="Coverage")
    axs[2].plot(moving_average(train_dict["max_fitness"]))
    axs[2].set(ylabel="Max Fitness")

    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True,
        help="Directory that contains the results.")
    args = parser.parse_args()

    print_stats(args.save_dir)
    plot(args.save_dir)