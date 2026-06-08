from nsga2_utils import NSGA2Utils
from population import Population
from tqdm import tqdm


class Evolution:

    def __init__(self, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9, crossover_param=2, mutation_param=5):
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param,
                                mutation_param)
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals

    def evolve(self):
        print('\n' + '=' * 70)
        print('开始NSGA-II优化...')
        print(f'种群大小: {self.num_of_individuals}, 迭代代数: {self.num_of_generations}')
        print(f'锦标赛参数: {self.utils.num_of_tour_particips}, 交叉参数: {self.utils.crossover_param}, 变异参数: {self.utils.mutation_param}')
        print('=' * 70)
        
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)
        returned_population = None
        
        print('\n开始进化迭代...')
        print('-' * 70)
        
        for i in tqdm(range(self.num_of_generations), desc='进化进度'):
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])
            returned_population = self.population
            self.population = new_population
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)
            
            # 每10代输出一次统计信息
            if (i + 1) % 10 == 0 or i == 0:
                pareto_front = self.population.fronts[0]
                if len(pareto_front) > 0:
                    obj_dim = len(pareto_front[0].objectives)
                    ranges = []
                    for m in range(obj_dim):
                        values = [ind.objectives[m] for ind in pareto_front]
                        ranges.append(f'Obj {m + 1} range: [{min(values):.6f}, {max(values):.6f}]')
                    tqdm.write(
                        f'代数 {i+1:4d}/{self.num_of_generations}, '
                        f'Pareto前沿: {len(pareto_front):3d}个解, '
                        + ', '.join(ranges)
                    )
        
        print('\n' + '-' * 70)
        print(f'优化完成! 最终Pareto前沿: {len(returned_population.fronts[0])}个解')
        print('=' * 70)
        
        return returned_population.fronts[0]
