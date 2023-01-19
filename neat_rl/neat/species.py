

class Species:
    def __init__(self, args, id=0):
        self.args = args
        self.orgs = []
        self.age = 0
        self.last_updated = 0
        self.species_id = id
        self.adj_fitness = 0
    
    def add(self, org):
        self.orgs.append(org)
    
    def first(self):
        return self.orgs[0]

    def update_age(self):
        self.age += 1
    
    @property
    def avg_fitness(self):
        return sum([o.avg_fitness for o in self.orgs]) / len(self.orgs)
