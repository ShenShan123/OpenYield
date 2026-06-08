from .individual import Individual
import random
import numpy as np

class Problem:

    def __init__(self, objectives, num_of_variables, variables_range, variables_type, expand=True, same_range=False, default_features=None):
        """
        variables_type: list of 'int' or 'float' indicating the type of each variable
        """
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.expand = expand
        self.variables_range = []
        self.variables_type = variables_type if not same_range else [variables_type[0]] * num_of_variables
        self.default_features = default_features
        self._first_individual = True
        
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range

    def generate_individual(self):
        individual = Individual()
        individual.features = []
        if self._first_individual and self.default_features is not None:
            self._first_individual = False
            individual.features = list(self.default_features)
            return individual
        for i in range(self.num_of_variables):
            if self.variables_type[i] == 'int':
                individual.features.append(random.randint(int(self.variables_range[i][0]), int(self.variables_range[i][1])))
            else:
                individual.features.append(random.uniform(*self.variables_range[i]))
        return individual

    def calculate_objectives(self, individual):
        if self.expand:
            individual.objectives = [f(*individual.features) for f in self.objectives]
        else:
            individual.objectives = [f(individual.features) for f in self.objectives]
