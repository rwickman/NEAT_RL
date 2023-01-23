import random

from neat_rl.neat.organism import Organism
from neat_rl.neat.species import Species
from neat_rl.neat.mutator import GradientMutator
from neat_rl.neat.reproduction import GradientReproduction

class GradientPopulation:
    def __init__(self, args, td3ga):
        self.args = args
        self.breeder = GradientReproduction(args)
        self.td3ga = td3ga
        self.org_id_to_species = {}
        self.species_list = []
        self.cur_id = 1
        self.generation = 0
        self.orgs = []
    
    def setup(self, net):
        self.base_org = Organism(self.args, net)
        self.orgs = self.spawn(self.base_org, self.args.pop_size)
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

    def _create_species(self):
        species = Species(self.args, len(self.species_list))
        self.species_list.append(species)
        return species

    def speciate(self):
        """Create the initial species and add organisms to them."""        
        # Create the species
        for _ in range(self.args.num_species):
            self._create_species()

        # Create organism for each species
        for i in range(self.args.pop_size):
            cur_species_id = i % self.args.num_species
            self.species_list[cur_species_id].add(self.orgs[i])


    def spawn(self, base_org, pop_size):
        orgs = []
        for i in range(pop_size):
            copy_org = base_org.copy(self.cur_id)
            self.cur_id += 1
            orgs.append(copy_org)
        
        return orgs

    def breed(self, cur_species):
        rand_int = random.randint(0, 1)
        if self.args.no_train_diversity:
            rand_int = random.randint(0, 1)
        else:
            rand_int = random.randint(0, 2)
        
        if self.args.only_pg:
            rand_int = 0

        if rand_int == 0:
            parent_1 = random.choice(cur_species.orgs)
            parent_2 = parent_1
            child_net = parent_1.net.copy(transfer_weights=True)
            self.td3ga.pg_update(child_net, cur_species.species_id)
        elif rand_int == 2:
            parent_1 = random.choice(cur_species.orgs)
            parent_2 = parent_1
            child_net = parent_1.net.copy(transfer_weights=True)
            self.td3ga.diversity_pg_update(child_net, cur_species.species_id)
        else:
            if len(cur_species.orgs) > 1:
                parent_1, parent_2 = random.sample(cur_species.orgs, 2)
            else:
                parent_1 = parent_2 = cur_species.orgs[0]


            child_net = self.breeder.reproduce(
                parent_1.net, parent_2.net)

        new_org = Organism(
            self.args, child_net, gen=max(parent_1.generation, parent_2.generation) + 1, id=self.cur_id)

        # Increment the current organism ID
        self.cur_id += 1
        return new_org
    
    def prune_species(self, cur_species):
        species_len = len(cur_species.orgs)

        # Randomize order so that sorting uniform avg fitness is random
        random.shuffle(cur_species.orgs)
        
        # Sort so best organisms are first
        if self.args.random_sort:
            pass
        elif self.args.diversity_bonus_sort:
            cur_species.orgs.sort(key=lambda x: x.bonus_avg, reverse=True) 
        elif self.args.best_diversity_sort:
            cur_species.orgs.sort(key=lambda x: x.bonus_best, reverse=True)
        else:
            cur_species.orgs.sort(key=lambda x: x.avg_fitness, reverse=True)

        # Calculate how many organsims should remain "alive"
        num_live = int(max(self.args.survival_rate * species_len, 1))

        # Remove all but the top
        cur_species.orgs = cur_species.orgs[:num_live]

        num_spawn = species_len - num_live
        return num_spawn

    def evolve(self):
        """Remove worst organisms and spawn organsism from breeding best organisms."""
        # Reset the organisms
        self.orgs = []

        # Create next iteration of organisms
        for cur_species in self.species_list:
            cur_species.update_age()
            num_spawn = self.prune_species(cur_species)

            for org in cur_species.orgs:
                org.age += 1
            
            # Save new orgs in list to prevent breeding with new_org
            new_orgs = []
            for _ in range(num_spawn):
                new_org = self.breed(cur_species)
                new_orgs.append(new_org)
            
            cur_species.orgs.extend(new_orgs)

            # Add species' organsims to list of all organisms
            self.orgs.extend(cur_species.orgs)

        self.org_id_to_species = {}
        for cur_species in self.species_list:
            for org in cur_species.orgs:
                self.org_id_to_species[org.id] = cur_species.species_id
        
        self.generation += 1

        assert len(self.orgs) == self.args.pop_size


    def get_best(self):
        best_fitness = best_org = None
        for org in self.orgs:
            if best_fitness is None or org.best_fitness > best_fitness:
                best_org = org
                best_fitness = org.best_fitness
        
        return best_org
                