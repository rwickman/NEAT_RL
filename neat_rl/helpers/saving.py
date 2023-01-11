import json, os
import torch

from neat_rl.neat.population import GradientPopulation
from neat.organism import Organism

def create_org_dict(org, prefix_dir):
    org_dict = {
        "id": org.id,
        "avg_fitness": org.avg_fitness,
        "generation": org.generation,
        "best_fitness": org.best_fitness,
        "num_updates": org._num_updates
    }
    model_file = os.path.join(prefix_dir, f"net_{org.id}.pt") 
    torch.save(org.net.state_dict(), model_file)
    org_dict["network"] = model_file

    return org_dict


def save_population(population, save_file):
    prefix_dir = save_file.split(".json")[0]
    if not os.path.exists(prefix_dir):
        os.mkdir(prefix_dir)
    else:
        # Remove all the old models
        for model_file in os.listdir(prefix_dir):
            if ".pt" in model_file:
                os.remove(os.path.join(prefix_dir, model_file))

    orgs = []
    for org in population.orgs:
        org_dict = create_org_dict(org, prefix_dir)
        orgs.append(org_dict)
    
    # Save the species list
    species_list = []
    for species in population.species_list:
        species_dict = {
            "id": species.species_id,
            "org_ids" : [org.id for org in species.orgs],
            "age": species.age
        }
        
        species_list.append(species_dict)

    # Save the base organism
    base_org = create_org_dict(population.base_org, prefix_dir)
    
    # Aggregate everything into this population dictionary
    pop_dict = {
        "orgs": orgs,
        "generation": population.generation,
        "cur_id": population.cur_id,
        "base_org": base_org,
        "species_list": species_list
    }

    with open(save_file, "w") as f:
        json.dump(pop_dict, f)
    
    return pop_dict

def _load_organism(args, org_dict, td3ga):
    net = td3ga.actor.copy(transfer_weights=False)
    net.load_state_dict(torch.load(org_dict["network"]))

    org = Organism(args, net, org_dict["generation"], org_dict["id"])
    if "best_fitness" in org_dict:
        org.best_fitness = org_dict["best_fitness"]

    if "avg_fitness" in org_dict:
        # Bit of a heuristic to assume avg_fitness is fitness sum
        org._num_updates = org_dict["num_updates"]
        org._fitness_avg = org_dict["avg_fitness"]

    return org

def load_population(args, td3ga):
    prefix_dir = args.save_file.split(".json")[0]
    # Load the population dictionary
    with open(args.save_file) as f:
        pop_dict = json.load(f)
    
    population = GradientPopulation(args, td3ga)
    population.cur_id = pop_dict["cur_id"]
    population.generation = pop_dict["generation"]
    population.base_org = _load_organism(args, pop_dict["base_org"], td3ga)
    
    # Load the organisms
    org_index = {} # Used to quickly retrieve organisms
    for org_dict in pop_dict["orgs"]:
        org = _load_organism(args, org_dict, td3ga)
        population.orgs.append(org)
        org_index[org.id] = org

    # Load the species
    for species_dict in pop_dict["species_list"]:
        species = population._create_species()
        species.age = species_dict["age"]

        # Add the organisms to the species
        for org_id in species_dict["org_ids"]:
            species.add(org_index[org_id])

    return population