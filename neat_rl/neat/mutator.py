from neat.mutator import Mutator

class GradientMutator(Mutator):
    def __init__(self, args, inv_counter):
        super().__init__(args, inv_counter)
    
    def mutate_add_node(self, net):
        return
    
    def mutate_add_link(self, net):
        return
    
    def mutate_link_weights(self, net):
        return