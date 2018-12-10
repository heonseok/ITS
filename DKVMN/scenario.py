import numpy as np
import os

from utils import *
from model import *

class BestScenario():
    def __init__(self, ordering, num_actions):

        self.ordering = ordering
        self.num_actions = num_actions

        self.permutation_fixed = np.random.permutation(self.num_actions)
        self.count = 0

        print('Best Scenario')


    def get_name(self):
        return 'best_' + self.ordering

    def get_action(self, state):
        if self.count % self.num_actions == 0:

            if self.ordering == 'ascending':
                self.ordering_list = range(self.num_actions)
            elif self.ordering == 'descending':
                self.ordering_list = range(self.num_actions-1,-1,-1)
            elif self.ordering == 'permutation':
                self.ordering_list = np.random.permutation(self.num_actions)
            elif self.ordering == 'permutation_fixed':
                self.ordering_list = self.permutation_fixed

            self.count = 0

        result = self.ordering_list[self.count]
        self.count += 1

        return result, 1

class RandomScenario():
    def __init__(self, num_actions, dkvmn):

        self.num_actions = num_actions
        self.correct_list = np.zeros(self.num_actions)
        self.dkvmn = dkvmn

        self.init_counter = self.dkvmn.get_init_counter()
        self.init_concept_counter = self.dkvmn.get_init_concept_counter()

        print('Random Scenario')

    def get_probability(self, value_matrix):
        return self.dkvmn.get_prediction_probability(value_matrix, self.init_counter, self.init_concept_counter)

    def get_name(self):
        return 'random'

    def get_action(self, state):
        ordering_list = np.random.permutation(self.num_actions)
        for action_idx in ordering_list:
           if self.correct_list[action_idx] == 0:
               break


        return action_idx, 1


class MaxProbScenario():
    def __init__(self, num_actions, dkvmn):
        self.num_actions = num_actions
        self.correct_list = np.zeros(self.num_actions)
        self.dkvmn = dkvmn

        self.init_counter = self.dkvmn.get_init_counter()
        self.init_concept_counter = self.dkvmn.get_init_concept_counter()

        print('MaxProb Scenario')

    def get_probability(self, value_matrix):
        return self.dkvmn.get_prediction_probability(value_matrix, self.init_counter, self.init_concept_counter)

    def get_name(self):
        return 'maxprob'

    def get_action(self, state):
        probs = self.get_probability(state)
        sorted_idx = np.argsort(-probs)
        for action_idx in sorted_idx:
            if self.correct_list[action_idx] == 0:
                break

        self.correct_list[action_idx] = 1
        return action_idx, 1

if __name__ == "__main__":
    best = BestScenario('ascending', 10)
    print(best.get_name())

    for i in range(30):
        print(best.get_action(None))


    random_scearnio = RandomScenario(10)
    print(random_scearnio.get_name())

    for i in range(30):
        print(random_scearnio.get_action(None))
