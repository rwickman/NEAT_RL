
class Organism:
    def __init__(self, args, net, gen=0, id=0):
        self.args = args
        self.net = net # The network controlling the behavior of the organism
        self.generation = gen # Tells what generation this organism is from
        self.fitness = 0
        self.best_fitness = -100000
        self.id = id
        self._fitness_avg = 0
        self._num_updates = 0
        self.age = 0
        self.behavior = None
        self.bonus_avg = 0
        self.bonus_best = 0

    def copy(self, org_id=0):
        copy_net = self.net.copy(transfer_weights=False)
        copy_org = Organism(self.args, copy_net, self.generation, org_id)
        copy_org.fitness = self.fitness
        copy_org.best_fitness = self.best_fitness
        copy_org._fitness_avg = self._fitness_avg
        copy_org._num_updates = self._num_updates
        copy_org.age = self.age
        copy_org.generation = self.generation
        return copy_org

    def update_fitness(self, fitness, diversity_bonus=None):
        #print("fitness", fitness, "diversity_bonus", diversity_bonus, "self.avg_fitness", self.avg_fitness, "self.bonus_avg", self.bonus_avg, "age", self.age)

        self._num_updates += 1
        # Calculation the moving average
        self._fitness_avg += (fitness - self._fitness_avg) / self._num_updates
        self.best_fitness = max(fitness, self.best_fitness)
        self.fitness = fitness
        if diversity_bonus:
            cur_reward = fitness + self.args.disc_lam * diversity_bonus
            #self.bonus_avg = self.avg_fitness + self.args.disc_lam * diversity_bonus
            #self.bonus_avg += (cur_reward - self.bonus_avg) / self._num_updates
            self.bonus_avg = diversity_bonus#(diversity_bonus - self.bonus_avg) / self._num_updates
            #print("AFTER self.avg_fitness", self.avg_fitness, "self.bonus_avg", self.bonus_avg, "\n")
            self.bonus_best = max(self.bonus_best, cur_reward)

    @property
    def avg_fitness(self):
        return self._fitness_avg 

    def __call__(self, x):
        return self.net(x)