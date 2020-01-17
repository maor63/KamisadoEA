import copy
import operator
import os
import random
import timeit
from _operator import attrgetter
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

from KamisadoGame.possible_moves_agent import PossibleMovesAgent
from KamisadoGame.possible_striking_agent import PossibleStrikingAgent
from KamisadoGame.striking_position_agent import StrikingPositionAgent
from KamisadoGame.tower_progress_agent import TowerProgressAgent
from KamisadoGame.kamisado import Kamisado, Player
from KamisadoGame.random_agent import RandomAgent
import pandas as pd

seed = 10
random.seed(seed)


def getBoard(board):
    return board


def if_then_else(condition, out1, out2):
    return out1 if condition else out2


def getOpenEndPostionsCount(board):
    assert isinstance(board, Kamisado)
    player = board.current_player
    tower_places = np.zeros((8, 8))
    if Player.WHITE == player:
        for pos in board.black_player_pos.values():
            tower_places[pos] = 1
        open_end_position = tower_places[:1].sum()
    else:
        for pos in board.white_player_pos.values():
            tower_places[pos] = 1
        open_end_position = tower_places[-1:].sum()
    return open_end_position


def getStrikingPositionFrec(board):
    assert isinstance(board, Kamisado)
    return strikingPossitionEval(board, board.current_player)


def getTowerProgressFrec(board):
    return towerProgressEval(board, board.current_player)


def getPossiblePossitionFrec(board):
    return possibleMovesEval(board, board.current_player)


def moveSrikingPositionCount(board, move_tuple):
    assert isinstance(board, Kamisado)
    if move_tuple:
        new_board = board.move_tower(*move_tuple)
        new_board.current_player = board.current_player
        return get_board_striking_count(new_board)
    else:
        return 0


def get_board_striking_count(new_board):
    possible_moves = new_board.getPossibleMovesTuples()
    possible_moves = [(tower, move) for (tower, move) in possible_moves if move is not None]
    return len(get_win_moves(possible_moves))


def getTowerPassHafe(board):
    assert isinstance(board, Kamisado)
    tower_places = np.zeros((8, 8))
    player = board.current_player
    player_positions = board.players_pos[player]
    for pos in player_positions.values():
        tower_places[pos] = 1
    if Player.WHITE == player:
        tower_progress = tower_places[:4].sum()
    else:
        tower_progress = tower_places[4:].sum()
    return tower_progress


def evalSideMove(board, move_tuple):
    assert isinstance(board, Kamisado)
    player = board.current_player
    if move_tuple[1] is None:
        move_tuple = (move_tuple[0], board.players_pos[player][move_tuple[0]])
    tower_y, tower_x = move_tuple[1]
    current_y, current_x = board.players_pos[player][move_tuple[0]]
    return tower_x - current_x


def isDiagonalMove(board, move_tuple):
    return evalSideMove(board, move_tuple) != 0


def getMoveDistanceFromStart(board, move_tuple):
    assert isinstance(board, Kamisado)
    player = board.current_player
    if move_tuple[1] is None:
        move_tuple = (move_tuple[0], board.players_pos[player][move_tuple[0]])
    tower, (tower_y, tower_x) = move_tuple
    if Player.WHITE == player:
        tower_progress_frec = (7 - tower_y)
    else:
        tower_progress_frec = tower_y
    return tower_progress_frec


def getMoveDistanceFromMid(board, move_tuple):
    assert isinstance(board, Kamisado)
    player = board.current_player
    if move_tuple[1] is None:
        move_tuple = (move_tuple[0], board.players_pos[player][move_tuple[0]])
    tower, (tower_y, tower_x) = move_tuple
    tower_progress_frec = abs(3 - tower_y)
    return tower_progress_frec


def isBlockStrikeMove(board, move_tuple):
    assert isinstance(board, Kamisado)
    new_board = board.clone()
    opponent = Player.WHITE if board.current_player != Player.WHITE else Player.BLACK
    new_board.current_player = opponent
    before = get_board_striking_count(new_board)

    new_board = board.move_tower(*move_tuple)
    new_board.current_player = opponent
    after = get_board_striking_count(board)
    return after < before


def isDoubleMove(board, move_tuple):
    assert isinstance(board, Kamisado)
    new_board = board.move_tower(*move_tuple)
    tower, move = new_board.getPossibleMovesTuples()[0]
    return move is None


def neighborBlockCount(board, move_tuple):
    assert isinstance(board, Kamisado)
    player = board.current_player
    if move_tuple[1] is None:
        move_tuple = (move_tuple[0], board.players_pos[player][move_tuple[0]])
    tower, (tower_y, tower_x) = move_tuple
    if Player.WHITE == player:
        front = [(tower_y - 1, tower_x + 1), (tower_y - 1, tower_x), (tower_y - 1, tower_x - 1)]
    else:
        front = [(tower_y + 1, tower_x + 1), (tower_y + 1, tower_x), (tower_y + 1, tower_x - 1)]
    blocks = sum(map(board.is_legal_move, front))
    return blocks


def evalBlockStrikeStatusOfMove(board, move_tuple):
    assert isinstance(board, Kamisado)
    new_board = board.clone()
    opponent = Player.WHITE if board.current_player != Player.WHITE else Player.BLACK
    new_board.current_player = opponent
    before = get_board_striking_count(new_board)

    new_board = board.move_tower(*move_tuple)
    new_board.current_player = opponent
    after = get_board_striking_count(board)
    return before - after


def getPossibleMovesCount(board):
    assert isinstance(board, Kamisado)
    return len(board.getPossibleMovesTuples())


def getEnemyPossibleMovesCount(board, move_tuple):
    assert isinstance(board, Kamisado)
    tower, pos = move_tuple
    new_board = board.move_tower(tower, pos)
    return getPossibleMovesCount(new_board)


def isThereWinMove(moves_tuples):
    if moves_tuples and None not in set(list(zip(*moves_tuples))[1]):
        return len(get_win_moves(moves_tuples)) > 0
    else:
        return False


def get_win_moves(moves_tuples):
    if moves_tuples:
        return [(tower, (y, x)) for (tower, (y, x)) in moves_tuples if y == 0 or y == 7]
    else:
        return []


def isLostMove(board, move_tuple):
    assert isinstance(board, Kamisado)
    if move_tuple:
        tower, move = move_tuple
        new_board = board.move_tower(tower, move)
        return True if isThereWinMove(new_board.getPossibleMovesTuples()) else False
    else:
        return 0


def isWinMove(move_tuple):
    if move_tuple:
        return True if isThereWinMove([move_tuple]) else False
    else:
        return False


def isWinOrLoseMove(board, move_tuple):
    assert isinstance(board, Kamisado)
    win_res = isWinMove(move_tuple)
    lose_res = isLostMove(board, move_tuple)
    return win_res + lose_res


random_player = RandomAgent()
tower_progress_agent = TowerProgressAgent(0)
striking_position_agent = StrikingPositionAgent(0)
possible_moves_agent = PossibleMovesAgent(0)
possible_striking_agent = PossibleStrikingAgent(0)


def towerProgressEval(board, max_player):
    assert isinstance(board, Kamisado)
    white_progress_sum = sum([7 - y for tower, (y, x) in board.white_player_pos.items()]) + 1
    black_progress_sum = sum([y - 0 for tower, (y, x) in board.black_player_pos.items()]) + 1
    if max_player == Player.WHITE:
        return white_progress_sum / black_progress_sum
    else:
        return black_progress_sum / white_progress_sum


def strikingPossitionEval(board, max_player):
    assert isinstance(board, Kamisado)
    new_board = board.clone()
    new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
    new_board.current_player = Player.WHITE
    white_striking_sum = len([1 for tower, pos in new_board.getPossibleMovesTuples() if pos and pos[0] == 0]) + 1

    new_board = board.clone()
    new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
    new_board.current_player = Player.BLACK
    black_striking_sum = len([1 for tower, pos in new_board.getPossibleMovesTuples() if pos and pos[0] == 7]) + 1
    if max_player == Player.WHITE:
        return white_striking_sum / black_striking_sum
    else:
        return black_striking_sum / white_striking_sum


def possibleMovesEval(board, max_player):
    assert isinstance(board, Kamisado)
    new_board = board.clone()
    new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
    new_board.current_player = Player.WHITE
    white_possible_sum = len(new_board.getPossibleMovesTuples()) + 1

    new_board = board.clone()
    new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
    new_board.current_player = Player.BLACK
    black_possible_sum = len(new_board.getPossibleMovesTuples()) + 1
    if max_player == Player.WHITE:
        return white_possible_sum / black_possible_sum
    else:
        return black_possible_sum / white_possible_sum


def get_gp_play_move(gp_policy):
    def gp_play_move(board):
        assert isinstance(board, Kamisado)
        moves_ranks = Counter(
            {move_tuple: gp_policy(board, move_tuple) for move_tuple in board.getPossibleMovesTuples()})
        selected_move = moves_ranks.most_common(1)[0][0]
        return selected_move

    return gp_play_move


def kamisado_simulator(p1_play_move, p2_play_move, max_steps_num=10000, init_board=None):
    board = Kamisado(init_board=init_board)
    players = [p1_play_move, p2_play_move]
    none_count = 0
    i = 0
    # while not board.is_game_won() and none_count < 10:
    start = timeit.default_timer()
    for i in range(max_steps_num):
        play_move = players[i % len(players)]
        move_tuple = play_move(board)
        move_tuple = move_tuple if move_tuple != () else random_player.play(board)
        tower, move = move_tuple
        board = board.move_tower(tower, move)
        none_count = none_count + 1 if not move else 0
        if board.is_game_won() or none_count >= 3:
            break
    end = timeit.default_timer()
    # print(f'game time {end-start} sec')
    return board, i


generations = 0


def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    global generations
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    generations += 1
    return chosen


def evalSolver(individual, games=50):
    start = timeit.default_timer()
    gp_policy = toolbox.compile(expr=individual)

    ea_play_move = get_gp_play_move(gp_policy)
    # print(individual)
    games_won = 0
    moves_counts = []
    tower_progress_list = []
    striking_position_list = []
    possible_moves_list = []
    possible_striking_list = []
    max_steps_num = 100
    random_board = list(random.sample(range(8), 8))

    for board_init in [None, [3, 5, 2, 6, 1, 7, 0, 4]]:
        d = min(generations // 100, 1)
        tower_progress_agent = TowerProgressAgent(d)
        striking_position_agent = StrikingPositionAgent(d)
        possible_moves_agent = PossibleMovesAgent(d)
        # possible_striking_agent = PossibleStrikingAgent(0)
        for agent in [random_player, tower_progress_agent, striking_position_agent, possible_moves_agent]:
            # for agent in [tower_progress_agent]:
            p2_play = agent.play
            board, moves_count = kamisado_simulator(ea_play_move, p2_play, max_steps_num, init_board=board_init)
            games_lost, games_tie, games_won1 = get_stats(board, max_steps_num, moves_count, moves_counts,
                                                          possible_moves_list, striking_position_list,
                                                          tower_progress_list, possible_striking_list,
                                                          Player.WHITE)

            board, moves_count = kamisado_simulator(p2_play, ea_play_move, max_steps_num, init_board=board_init)
            res = board.is_game_won()
            games_lost, games_tie, games_won2 = get_stats(board, max_steps_num, moves_count, moves_counts,
                                                          possible_moves_list, striking_position_list,
                                                          tower_progress_list, possible_striking_list,
                                                          Player.BLACK)
            games_won += games_won1 + games_won2

    end = timeit.default_timer()
    # print(f'time {end - start} sec')
    tree_length = len(individual)
    depth_mult = 0.8
    return games_won * (1 + depth_mult * d), np.mean(moves_counts) * (1 + depth_mult * d), np.mean(
        tower_progress_list) * (
                   1 + depth_mult * d), np.mean(striking_position_list) * (1 + depth_mult * d)
    # depth_mult = 0.8
    # return games_won * (1 + depth_mult * d), np.mean(moves_counts) * (1 + depth_mult * d)
    # return games_won, np.mean(moves_counts)


def get_stats(board, max_steps_num, moves_count, moves_counts, possible_moves_list, striking_position_list,
              tower_progress_list, possible_striking_list, max_player=Player.WHITE):
    games_won = 0
    games_lost = 0
    games_tie = 0
    res = board.is_game_won()
    if res == max_player:
        games_won += 1
        moves_counts.append(2.3 - moves_count / max_steps_num)

    elif res is None:
        games_tie += 1
        moves_counts.append(1 + moves_count / max_steps_num)
    else:
        games_lost += 1
        moves_counts.append(1 + moves_count / max_steps_num)
    tower_progress_list.append(1 + towerProgressEval(board, max_player))
    striking_position_list.append(1 + strikingPossitionEval(board, max_player))
    possible_moves_list.append(1 + possibleMovesEval(board, max_player))
    # possible_striking_list.append(possible_striking_agent.evaluate_game(board, max_player))
    return games_lost, games_tie, games_won


def getMove(move_tuple):
    return move_tuple


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def selAgentTournament(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        aspirants = random.sample(individuals, tournsize)
        while len(aspirants) > 1:
            random.shuffle(aspirants)
            p1, p2 = aspirants.pop(), aspirants.pop()
            p1_move = get_playe_move_from_policy(p1)
            p2_move = get_playe_move_from_policy(p2)
            board, moves_count = kamisado_simulator(p1_move, p2_move)
            res = board.is_game_won()
            if res == Player.WHITE:
                aspirants.append(p1)
            else:
                aspirants.append(p2)
        chosen += aspirants
    return chosen


def get_playe_move_from_policy(p1):
    gp_policy = toolbox.compile(expr=p1)
    p1_move = get_gp_play_move(gp_policy)
    return p1_move


def get_score_for_two_players(p1_move, p2_move, games_count=100, init_board=None):
    score = np.array([0, 0, 0])
    for i in range(games_count):
        board, moves_count = kamisado_simulator(p1_move, p2_move, init_board=init_board)
        res = board.is_game_won()
        if res == Player.WHITE:
            score[0] += 1
        elif res == Player.BLACK:
            score[2] += 1
        else:
            score[1] += 1

        board, moves_count = kamisado_simulator(p2_move, p1_move, init_board=init_board)
        res = board.is_game_won()
        if res == Player.WHITE:
            score[2] += 1
        elif res == Player.BLACK:
            score[0] += 1
        else:
            score[1] += 1
    return score


def moveTower(board, move_tuple):
    assert isinstance(board, Kamisado)
    if move_tuple:
        tower, move = move_tuple
        possible_moves = board.get_possible_moves()
        if tower in possible_moves and move in possible_moves[tower]:
            return board.move_tower(*move_tuple)
        else:
            return board
    else:
        return board


def getCurrentPlayer(board):
    assert isinstance(board, Kamisado)
    return board.current_player.value


def getOtherPlayer(board):
    assert isinstance(board, Kamisado)
    return Player.WHITE.value if board.current_player != Player.WHITE else Player.BLACK.value


def tower_progress_eval(board, player):
    return tower_progress_agent.evaluate_game(board, player)


def striking_position_eval(board, player):
    return striking_position_agent.evaluate_game(board, player)


def possible_position_eval(board, player):
    return possible_moves_agent.evaluate_game(board, player)


pset = gp.PrimitiveSetTyped("main", [Kamisado, tuple], float)
pset.addPrimitive(getBoard, [Kamisado], Kamisado)
pset.addPrimitive(moveTower, [Kamisado, tuple], Kamisado)
pset.addPrimitive(getMoveDistanceFromStart, [Kamisado, tuple], float)
pset.addPrimitive(getMoveDistanceFromMid, [Kamisado, tuple], float)
pset.addPrimitive(neighborBlockCount, [Kamisado, tuple], float)
pset.addPrimitive(evalSideMove, [Kamisado, tuple], float)
pset.addPrimitive(isDoubleMove, [Kamisado, tuple], bool)
pset.addPrimitive(isDiagonalMove, [Kamisado, tuple], bool)
pset.addPrimitive(moveSrikingPositionCount, [Kamisado, tuple], float)
pset.addPrimitive(getEnemyPossibleMovesCount, [Kamisado, tuple], float)
pset.addPrimitive(evalBlockStrikeStatusOfMove, [Kamisado, tuple], float)
pset.addPrimitive(isBlockStrikeMove, [Kamisado, tuple], bool)
pset.addPrimitive(isWinOrLoseMove, [Kamisado, tuple], float)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(isLostMove, [Kamisado, tuple], bool)
pset.addPrimitive(isWinMove, [tuple], bool)
pset.addPrimitive(getMove, [tuple], tuple)
pset.addPrimitive(if_then_else, [bool, float, float], float)

pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)
# pset.addTerminal(Player.WHITE.value, Player)
# pset.addTerminal(Player.BLACK.value, Player)
pset.addTerminal([], list)
# pset.addTerminal((), tuple)
for i in range(11):
    pset.addTerminal(-i, float)
    pset.addTerminal(i, float)
pset.addTerminal(100, float)
pset.addTerminal(-100, float)
pset.renameArguments(ARG0="Board")
pset.renameArguments(ARG1="move_tuple")
# pset.addTerminal(Kamisado(), Kamisado)

creator.create("FitnessMax", base.Fitness, weights=(1.0, 2.0, 1.0, 1.0))
# creator.create("FitnessMax", base.Fitness, weights=(10.0, 0.5))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

max_tree_length = 7
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=max_tree_length)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSolver)
# toolbox.register("select", selAgentTournament, tournsize=5)
toolbox.register("select", selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=1, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_length))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_length))

games_won = tools.Statistics(lambda ind: ind.fitness.values[0])
# [tower_progress_sum, striking_position_sum, possible_moves_sum]
move_count_mean = tools.Statistics(lambda ind: ind.fitness.values[1])
progress_mean = tools.Statistics(lambda ind: ind.fitness.values[2])
strike_mean = tools.Statistics(lambda ind: ind.fitness.values[3])
# possible_moves_mean = tools.Statistics(lambda ind: ind.fitness.values[4])
# striking_possible_mean = tools.Statistics(lambda ind: ind.fitness.values[5])
mstats = tools.MultiStatistics(games_won=games_won, move_count_mean=move_count_mean, progress_mean=progress_mean,
                               strike_mean=strike_mean,
                               )
# mstats = tools.MultiStatistics(games_won=games_won, move_count_mean=move_count_mean)
# mstats = wins_stats
mstats.register("Avg", np.mean)
mstats.register("Std", np.std)
mstats.register("Median", np.median)
mstats.register("Min", np.min)
mstats.register("Max", np.max)

pop_size = 50
pop = toolbox.population(n=pop_size)
hof = tools.HallOfFame(1)

games_count = 1
cxpb = 0.7
mutpb = 0.05
ngen = 400
experiment_name = f'pop{pop_size}_gen{ngen}_cxpb{cxpb}_mutpb{mutpb}_max{max_tree_length}_increase_level'
print(experiment_name)
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats=mstats,
                                   halloffame=hof, verbose=True)
print(hof[0])

output_path = 'data/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(40, 8))
try:
    for idx, statistics in enumerate(
            ['games_won', 'move_count_mean']):
        gen = logbook.select("gen")
        fit_mins = logbook.chapters[statistics].select("Min")
        fit_avgs = logbook.chapters[statistics].select("Avg")
        fit_maxs = logbook.chapters[statistics].select("Max")
        fit_medians = logbook.chapters[statistics].select("Median")
        ax[idx].plot(gen, fit_mins, "b-", label="Minimum Fitness")
        ax[idx].plot(gen, fit_avgs, "r-", label="Average Fitness")
        ax[idx].plot(gen, fit_maxs, "g-", label="Max Fitness")
        ax[idx].plot(gen, fit_medians, "y-", label="Median Fitness")
        ax[idx].set_xlabel("Generation", fontsize=18)
        ax[idx].tick_params(labelsize=16)
        ax[idx].set_ylabel(f"{statistics}", color="b", fontsize=18)
        ax[idx].legend(loc="lower right", fontsize=14)
        ax[idx].set_title(f'Kamisado agents performance on {statistics}', fontsize=18)


except Exception as e:
    print(e)
plt.show()
plt.savefig(f'{experiment_name}.png', dpi=fig.dpi)

# exit(1)
p1_move = get_playe_move_from_policy(hof[0])
print(evalSolver(hof[0]))

cols = ['p1', 'p2', 'win', 'tie', 'lose']
train_rows = []
print('############################Train Data###################################')
# , [0, 2, 4, 6, 1, 3, 5, 7], [1, 3, 5, 7, 0, 2, 4, 6]
for board_init in [None, [3, 5, 2, 6, 1, 7, 0, 4]]:
    print(board_init)
    row = []
    score = get_score_for_two_players(p1_move, p1_move, games_count, init_board=board_init)
    print(f'p1 VS p1 score: {score}')
    row.append(score)

    score = get_score_for_two_players(p1_move, random_player.play, games_count, init_board=board_init)
    print(f'p1 VS random_player score: {score}')
    row.append(score)

    for i in range(4):
        tower_progress_agent = TowerProgressAgent(i)
        striking_position_agent = StrikingPositionAgent(i)
        possible_moves_agent = PossibleMovesAgent(i)
        score = get_score_for_two_players(p1_move, tower_progress_agent.play, games_count, init_board=board_init)
        print(f'p1 VS tower_progress_agent score: {score}, minmax depth {i}')
        row.append(score)

        score = get_score_for_two_players(p1_move, striking_position_agent.play, games_count, init_board=board_init)
        print(f'p1 VS striking_position_agent score: {score}, minmax depth {i}')
        row.append(score)

        score = get_score_for_two_players(p1_move, possible_moves_agent.play, games_count, init_board=board_init)
        print(f'p1 VS possible_moves_agent score: {score}, minmax depth {i}')
        row.append(score)

    train_rows.append(row)

spacial_agents = [[f'tower_progress_d{i}', f'striking_position_d{i}', f'possible_moves_d{i}'] for i in range(4)]
agents_names = ['p1', 'random'] + list(chain(*spacial_agents))

train_df_rows = []
for idx, score_list in enumerate(zip(*train_rows)):
    score_sum = np.array(score_list).sum(axis=0)
    train_df_rows.append(['p1', agents_names[idx], score_sum[0], score_sum[1], score_sum[2]])

train_df = pd.DataFrame(train_df_rows, columns=cols)
file_name = f'train_results_pop{pop_size}_gen{ngen}_cxpb{cxpb}_mutpb{mutpb}_max{max_tree_length}.csv'
train_df.to_csv(os.path.join(output_path, file_name))

test_rows = []
print('############################Test Data###################################')
for j in range(100):
    row = []
    init_board = list(range(8))
    random.shuffle(init_board)
    print(f'board init {init_board}')
    score = get_score_for_two_players(p1_move, p1_move, games_count, init_board=init_board)
    print(f'p1 VS p1 score: {score}')
    row.append(score)

    score = get_score_for_two_players(p1_move, random_player.play, games_count, init_board=init_board)
    print(f'p1 VS random_player score: {score}')
    row.append(score)

    for i in range(4):
        tower_progress_agent = TowerProgressAgent(i)
        striking_position_agent = StrikingPositionAgent(i)
        possible_moves_agent = PossibleMovesAgent(i)
        score = get_score_for_two_players(p1_move, tower_progress_agent.play, games_count, init_board=init_board)
        print(f'p1 VS tower_progress_agent score: {score}, minmax depth {i}')
        row.append(score)

        score = get_score_for_two_players(p1_move, striking_position_agent.play, games_count, init_board=init_board)
        print(f'p1 VS striking_position_agent score: {score}, minmax depth {i}')
        row.append(score)

        score = get_score_for_two_players(p1_move, possible_moves_agent.play, games_count, init_board=init_board)
        print(f'p1 VS possible_moves_agent score: {score}, minmax depth {i}')
        row.append(score)
    test_rows.append(row)

test_df_rows = []
for idx, score_list in enumerate(zip(*test_rows)):
    score_sum = np.array(score_list).sum(axis=0)
    test_df_rows.append(['p1', agents_names[idx], score_sum[0], score_sum[1], score_sum[2]])

test_df = pd.DataFrame(test_df_rows, columns=cols)
file_name = f'test_results_pop{pop_size}_gen{ngen}_cxpb{cxpb}_mutpb{mutpb}_max{max_tree_length}.csv'
test_df.to_csv(os.path.join(output_path, file_name))
