import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import pandas as pd
import numpy as np

sns.set(style="darkgrid")

benchmark_results_dirs = [
    "results_PGA/results_MAP-Elites-ES",
    "results_PGA/results_QDRL",
    "results_PGA/results_TD3",
    "results_PGA/results_PGA-MAP-Elites"
]

my_results = "all_trained_models/walker"
env = "Ant"
eval_nums = [20000, 40000, 60000, 80000, 100000]

def get_my_stats(my_results):    
    my_results_dirs = [os.path.join(my_results, d) for d in os.listdir(my_results)]
    fitness_vals = [[] for _ in range(len(eval_nums))]
    qd_vals = [[] for _ in range(len(eval_nums))]

    for d in my_results_dirs:
        train_dict_file = [os.path.join(d, f) for f in os.listdir(d) if "train_dict" in f][0]
        with open(train_dict_file) as f:
            train_dict = json.load(f)


        for i in range(len(eval_nums)):
            cur_idx = (eval_nums[i] - 1) // 64
            if i == len(eval_nums) - 1:
                cur_idx = len(train_dict["max_fitness"]) - 1

            fitness_vals[i].append(train_dict["max_fitness"][cur_idx])
            qd_vals[i].append(train_dict["total_fitness_archive"][cur_idx])


    fitness_avg = []
    qd_avg = []
    fitness_std = []
    qd_std = []
    for i in range(len(eval_nums)):
        fitness_avg.append(sum(fitness_vals[i]) / len(fitness_vals[i]))
        qd_avg.append(sum(qd_vals[i]) / len(qd_vals[i]))
        fitness_std.append(np.array(fitness_vals[i]).std())
        qd_std.append(np.array(qd_vals[i]).std())
        
    
    return fitness_avg, qd_avg, fitness_std, qd_std


def get_their_stats(root_dir, eval_num):
    results_dirs = os.listdir(root_dir)


    fitness_vals = [[] for _ in range(eval_num)]
    qd_vals = [[] for _ in range(eval_num)]
    
    for path_dir in results_dirs:
        if ".DS_Store" in path_dir:
            continue
        results_path = os.path.join(root_dir, path_dir)
        csv_file = [p for p in os.listdir(results_path) if ".csv" in p][0]
        csv_path = os.path.join(results_path, csv_file)
        if ".swp" in csv_path:
            continue
        df = pd.read_csv(csv_path)
        
        
        for i in range(eval_num):
            fitness_vals[i].append(df["max_fitness"][i])
            qd_vals[i].append(df["sum_fitness"][i])

                

    fitness_avg = []
    qd_avg = []
    fitness_std = []
    qd_std = []
    
    for i in range(eval_num):
        fitness_avg.append(sum(fitness_vals[i]) / len(fitness_vals[i]))
        qd_avg.append(sum(qd_vals[i]) / len(qd_vals[i]))
        fitness_std.append(np.array(fitness_vals[i]).std())
        qd_std.append(np.array(qd_vals[i]).std())
        


    

    return fitness_avg, qd_avg, fitness_std, qd_std


def shade_std(mean_vals, std_vals, ax):
    plots = [r for r in ax.get_children() if type(r)==Line2D]
    line_color = plots[-1].get_color()
    for i in range(len(eval_nums) - 1):
        ax.fill_between(
            np.linspace(eval_nums[i], eval_nums[i+1]),
            np.linspace(mean_vals[i] - std_vals[i], mean_vals[i+1] - std_vals[i + 1]), # Shade lower bound
            np.linspace(mean_vals[i] + std_vals[i], mean_vals[i+1] + std_vals[i+1]), # Shade upper bound
            alpha=0.075,
            color=line_color
        )

def plot(avg_fitness, qd_scores, fitness_std, qd_std):
    fig, axs = plt.subplots(2)
    axs[0].set(ylabel="Max-fitness")
    axs[1].set(ylabel="QD-score")

    x_vals = [20000, 40000, 60000, 80000, 100000]
    for i in range(len(avg_fitness)):
        sns.lineplot(ax=axs[0], x=x_vals, y=avg_fitness[i])
        sns.lineplot(ax=axs[1], x=x_vals, y=qd_scores[i])
        
        
        # Show the standard deviation shaded region
        shade_std(avg_fitness[i], fitness_std[i], axs[0])
        shade_std(qd_scores[i], qd_std[i], axs[1])

            
    

    plt.show()


avg_fitness_list = []
qd_score_list = []
fitness_std_list = []
qd_score_std_list = []

for d in benchmark_results_dirs:
    method = d.split("_")[-1]
    d = os.path.join(d, f"results_{env}_{method}")

    avg_fitness, qd_score, fitness_std, qd_std = get_their_stats(d, 5)
    avg_fitness_list.append(avg_fitness)
    qd_score_list.append(qd_score)

    fitness_std_list.append(fitness_std)
    qd_score_std_list.append(qd_std)

fitness, qd_scores, fitness_std, qd_std = get_my_stats(my_results)
avg_fitness_list.append(fitness)
qd_score_list.append(qd_scores)
fitness_std_list.append(fitness_std)
qd_score_std_list.append(qd_std)

plot(avg_fitness_list, qd_score_list, fitness_std_list, qd_score_std_list)

# print(len(fitness), len(qd_scores))
    # print(d, get_their_stats(d, 5))