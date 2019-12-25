import copy
import operator
import random
import timeit
from collections import Counter, defaultdict
import numpy
from itertools import chain, combinations
from functools import partial
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.tools import selRandom

from KamisadoGame.kamisado import Kamisado, Player
from KamisadoGame.random_agent import RandomAgent

pset = gp.PrimitiveSetTyped("main", [Kamisado, tuple], float)
pset.addPrimitive(getBoard, [Kamisado], Kamisado)
pset.addPrimitive(getTowerProgress, [Kamisado, tuple], float)
pset.addPrimitive(getPossibleMovesCount, [Kamisado, tuple], float)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(getMedian, [list], tuple)
pset.addPrimitive(isLostMove, [Kamisado, tuple], bool)
pset.addPrimitive(isWinMove, [tuple], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)
pset.addTerminal([], list)
pset.addTerminal((), tuple)
pset.renameArguments(ARG0="Board")
# pset.addTerminal(Kamisado(), Kamisado)

creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
max_ = 5
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSolver)
# toolbox.register("select", selAgentTournament, tournsize=5)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=2, max_=max_)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_))

wins_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
move_count_stats = tools.Statistics(lambda ind: ind.fitness.values[1])
mstats = tools.MultiStatistics(wins_stats=wins_stats, move_count_stats=move_count_stats)
# mstats = wins_stats
mstats.register("Avg", np.mean)
mstats.register("Std", np.std)
mstats.register("Median", np.median)
mstats.register("Min", np.min)
mstats.register("Max", np.max)

pop = toolbox.population(n=10)
hof = tools.HallOfFame(1)

pop, logbook = algorithms.eaSimple(pop, toolbox, 0.7, 0.01, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
print(hof[0])
