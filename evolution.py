import copy
import os
from player import Player
import numpy as np
from operator import attrgetter

class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # top-k algorithm implementation
        players.sort(key=lambda x: x.fitness, reverse=True)

        # Q-tournament algorithm implementation
        #players = self.q_tournament(players ,num_players ,4)
        
        # roulette wheel implementation
        #players = self.roulette_wheel(players ,num_players)
   
        # to plot learning curve
        self.save_fitness(players)

        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            #prev_players = self.q_tournament(prev_players ,num_players ,3)
            new_players = self.apply_crossover(prev_players)

            for child in new_players:
                self.mutate(child)
            # new_players = prev_players
            return new_players

    

    def apply_crossover(self, prev_players):
        new_players = []

        for i in range(0, len(prev_players), 2):
            i1 = prev_players[i]
            i2 = prev_players[i+1]

            new_child1 = self.clone_player(i1)
            new_child2 = self.clone_player(i2)

            for i in range(len(new_child1.nn.w)):
                shape = new_child1.nn.w[i].shape

                new_child1.nn.w[i][:, int(shape[1]/2):] = i2.nn.w[i][:, int(shape[1]/2):]
                new_child2.nn.w[i][:, int(shape[1] / 2):] = i1.nn.w[i][:, int(shape[1]/ 2):]

            for i in range(len(new_child1.nn.b)):
                shape = new_child1.nn.w[i].shape

                new_child1.nn.b[i][:, int(shape[1] / 2):] = i2.nn.b[i][:, int(shape[1] / 2):]
                new_child2.nn.b[i][:, int(shape[1] / 2):] = i1.nn.b[i][:, int(shape[1] / 2):]

            new_players.append(new_child1)
            new_players.append(new_child2)

        return new_players

    def mutate(self, child):
        mutation_threshold = 0.7
        center = 0
        margin = 0.3

        for i in range(len(child.nn.w)):
            if np.random.random_sample() >= mutation_threshold:
                child.nn.w[i] += np.random.normal(center, margin, size=(child.nn.w[i].shape))

        for i in range(len(child.nn.b)):
            if np.random.random_sample() >= mutation_threshold:
                child.nn.b[i] += np.random.normal(center, margin, size=(child.nn.b[i].shape))

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def save_fitness(self, players):
        if not os.path.exists('fitness'):
            os.makedirs('fitness')

        f = open("fitness/output1.txt", "a")
        for p in players:
            f.write(str(p.fitness))
            f.write(" ")
        f.write("\n")
        f.close()
        
    def q_tournament(self ,players ,num_players ,q ):
        selected = []
        for i in range(num_players) :
             q_selections = np.random.choice(players, q)
             selected.append(max(q_selections, key=attrgetter('fitness')))

        return selected

    def roulette_wheel(self ,players ,num_player):
        next_generation = []
        fit_sum = sum([player.fitness for player in players])
        probs = [player.fitness/fit_sum for player in players]
        for i in range(num_player):
            p = np.random.choice(players, 1, p=probs)
            next_generation.append(p)

        return next_generation