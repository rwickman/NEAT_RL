import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from neat_rl.helpers.util import load_archive, load_kdtree
import pickle

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
my_cmap = mpl.cm.viridis

def print_stats(save_dir, env):
    
    pop_file = os.path.join(save_dir, "pop.json")
    if env:    
        train_dict_file = os.path.join(save_dir, f"train_dict_{env}.json")
    else:
        train_dict_file = os.path.join(save_dir, f"train_dict.json")

    with open(pop_file) as f:
        pop_dict = json.load(f)

    with open(train_dict_file) as f:
        train_dict = json.load(f)
    
    org_id_to_species = pop_dict["org_id_to_species"]
    avg_behavior = {}
    species_sizes = {}
    
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
        species_id = org_id_to_species[str(org_dict["id"])]
        
        # Add to the behavior sum
        if species_id not in avg_behavior:
            avg_behavior[species_id] = np.array(org_dict["behavior"])  
            species_sizes[species_id] = 1
        else:
            avg_behavior[species_id] += np.array(org_dict["behavior"])  
            species_sizes[species_id] += 1
    
    # Average out the behavior of each species
    for k in range(len(avg_behavior)):
        avg_behavior[k] = avg_behavior[k] / species_sizes[k]
        print(f"SPECIES {k} AVG BEHAVIOR {avg_behavior[k]}")

    print("TOTAL GENERATIONS:", pop_dict["generation"])
    print("MAX AGE:", max_age)
    print("AVG AGE: ", avg_age/len(pop_dict["orgs"]))
    print("MAX GENERATION:", max_generation)
    print("MIN GENERATION:", min_generation)
    print("AVG GENERATION:", avg_generation/len(pop_dict["orgs"]))
    print("COVERAGE LAST 10:", train_dict["coverage"][-10:])
    print("total_fitness_archive", train_dict["total_fitness_archive"][-5:])
    print("MAX FITNESS:", train_dict["max_fitness"][-1])

def plot_species(save_dir):
    pop_file = os.path.join(save_dir, "pop.json")    
    with open(pop_file) as f:
        pop_dict = json.load(f)
    
    org_id_to_species = pop_dict["org_id_to_species"]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    cmap = mpl.colormaps["tab20"]
    norm = mpl.colors.Normalize(vmin=0, vmax=max(org_id_to_species.values()))
    for org_dict in pop_dict["orgs"]:
        species_id = org_id_to_species[str(org_dict["id"])]
        if len(org_dict["behavior"]) == 2:
            ax.scatter(org_dict["behavior"][0], org_dict["behavior"][1], color=cmap(norm(species_id)))
        else:
            ax.scatter(org_dict["behavior"][0], 0.5, color=cmap(norm(species_id)))

    plt.show()


def print_archive(save_dir, env):
    args_file = os.path.join(save_dir, "args.json")    
    with open(args_file) as f:
        args = json.load(f)
    
    archive_file = os.path.join(save_dir, f"archive_{env}.json")
    if not os.path.exists(archive_file):
        archive_file = os.path.join(save_dir, "archive.json")

    archive, archive_species_ids = load_archive(archive_file)
    kdt = load_kdtree(env)
    kdt_points = np.array(kdt.data)
    archive_points = list(archive.keys())
    num_missing = 0
    min_fit = min(list(archive.values()))
    max_fit = max(list(archive.values()))
    print(min_fit, max_fit)
    norm = mpl.colors.Normalize(vmin=min_fit, vmax=max_fit)

    cmap = mpl.colormaps["tab20"]
    norm_2 = mpl.colors.Normalize(vmin=0, vmax=args["num_species"])

    # Set plot params
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(2):

        ax[i].set_xlim([0, 1])
        ax[i].set_ylim([0, 1])
    
    
    for point in archive_points:
        if len(point) == 2:
            ax[0].scatter(point[0], point[1], color=my_cmap(norm(archive[point])))
            ax[1].scatter(point[0], point[1], color=cmap(norm_2(archive_species_ids[point])))
        else:
            ax[0].scatter(point[0], 0.5, color=my_cmap(norm(archive[point])))
            ax[1].scatter(point[0], 0.5, color=cmap(norm_2(archive_species_ids[point])))

    plt.show()
    # params = {
    #     "axes.labelsize": 18,
    #     "legend.fontsize": 18,
    #     "xtick.labelsize": 18,
    #     "ytick.labelsize": 18,
    #     "text.usetex": False,
    #     "figure.figsize": [6, 6],
    # }
    # mpl.rcParams.update(params)
    # min_fit = min(list(archive.values()))
    # max_fit = max(list(archive.values()))

    # fig, ax = plt.subplots(facecolor="white", edgecolor="white")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set(adjustable="box", aspect="equal")
    # norm = mpl.colors.Normalize(vmin=min_fit, vmax=max_fit)

    # bd_axis=["Leg 1 proportion of contact-time", "Leg 2 proportion of contact-time"]

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlabel(bd_axis[0])
    # ax.set_ylabel(bd_axis[1])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax)
    # cbar.set_label("Fitness", size=24, labelpad=5)
    # cbar.ax.tick_params(labelsize=18)
    # plt.show()
    # for point in kdt_points:
    #     if tuple(point.tolist()) not in archive_points:
    #         print(point)
    #         num_missing += 1
    # print(num_missing)

    # print(archive)



def plot(save_dir, env=None):
    if env:
        train_dict_file = os.path.join(save_dir, f"train_dict_{env}.json")
    else:
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
    parser.add_argument("--env",
        help="Name of the environment.")
    
    args = parser.parse_args()

    print_stats(args.save_dir, args.env)
    # if args.env:
    #     plot_species(args.save_dir)
    #     print_archive(args.save_dir, args.env)
    plot(args.save_dir, args.env)