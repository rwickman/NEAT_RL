import random

from neat.organism import Organism
from neat.population import Population
from neat.organism import Organism

from neat_rl.neat.mutator import GradientMutator
from neat_rl.neat.reproduction import GradientReproduction

class GradientPopulation(Population):
    def __init__(self, args, td3ga):
        super().__init__(args)
        self.mutator = GradientMutator(args, self.inv_counter)
        self.breeder = GradientReproduction(args)
        self.td3ga = td3ga
        self.org_id_to_species = {}

    
    def setup(self, net):
        self.base_org = Organism(self.args, net)
        self.orgs = self.spawn(self.base_org, self.args.init_pop_size)
        self.speciate()
        self.org_id_to_species = {}
        total_orgs = 0
        for cur_species in self.species_list:
            total_orgs += len(cur_species.orgs)
            for org in cur_species.orgs:

                assert org.id not in self.org_id_to_species 
                self.org_id_to_species[org.id] = cur_species.species_id
        print("total_orgs", total_orgs, len(self.orgs))
        assert total_orgs == len(self.orgs)

        
    def spawn(self, base_org, pop_size):
        orgs = []
        for i in range(pop_size):
            copy_org = base_org.copy(self.cur_id)
            self.cur_id += 1
            copy_org.net = base_org.net.copy(transfer_weights = False)
            orgs.append(copy_org)
        
        return orgs
    
    def speciate_fn(self, org_1, org_2):
        total_diff = 0
        for net_1_param, net_2_param in zip(org_1.net.parameters(), org_2.net.parameters()):
            total_diff += (net_1_param.data - net_2_param).sum()

        return total_diff.item()
    
    def breed(self, cur_species):
        parent_1 = random.choice(cur_species.orgs)
        if random.random() <= self.args.pg_rate:
            parent_2 = parent_1
            child_net = parent_1.net.copy()
            self.td3ga.pg_update(child_net, cur_species.species_id)
        else:
            if random.random() <= self.args.mutate_no_crossover:
                parent_2 = parent_1
            else:
                if random.random() <= self.args.reproduce_interspecies_rate:
                    # Choose a random organism across all the species
                    parent_2 = random.choice(random.choice(self.species_list).orgs)
                else:
                    parent_2 = random.choice(cur_species.orgs)
            
            child_net = self.breeder.reproduce(
                parent_1.net, parent_2.net)

            # Mutate the genotype
            self.mutate_child(child_net)
            

        new_org = Organism(
            self.args, child_net, gen=max(parent_1.generation, parent_2.generation) + 1, id=self.cur_id)

        # Increment the current organism ID
        self.cur_id += 1
        return new_org, parent_1, parent_2
    
    def evolve(self):
        super().evolve()
        self.org_id_to_species = {}
        for cur_species in self.species_list:
            for org in cur_species.orgs:
                self.org_id_to_species[org.id] = cur_species.species_id
