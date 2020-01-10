import copy
import operator
import os
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

from KamisadoGame.possible_moves_agent import PossibleMovesAgent
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


def moveSrikingPositionCount(board, move_tuple):
    assert isinstance(board, Kamisado)
    if move_tuple:
        new_board = board.move_tower(*move_tuple)
        new_board.current_player = board.current_player
        possible_moves = getPossibleMoves(new_board)
        possible_moves = [(tower, move) for (tower, move) in possible_moves if move is not None]
        return len(get_win_moves(possible_moves))
    else:
        return 0


def getTotalTowerProgress(board):
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


def getMoveTowerProgress(board, move_tuple):
    assert isinstance(board, Kamisado)
    tower_places = np.zeros((8, 8))
    player = board.current_player
    if move_tuple[1] is None:
        move_tuple = (move_tuple[0], board.players_pos[player][move_tuple[0]])
    tower, (tower_y, tower_x) = move_tuple
    if Player.WHITE == player:
        tower_progress_frec = (7 - tower_y)
    else:
        tower_progress_frec = tower_y
    return tower_progress_frec


def getPossibleMovesCount(board):
    assert isinstance(board, Kamisado)
    return len(getPossibleMoves(board))


def getPossibleMoves(board):
    assert isinstance(board, Kamisado)
    player_possible_moves = board.get_possible_moves()
    possible_moves = []
    for tower, moves in player_possible_moves.items():
        for move in moves:
            possible_moves.append((tower, move))
    return possible_moves


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
        return -1000 if isThereWinMove(getPossibleMoves(new_board)) else 0
    else:
        return 0


def isWinMove(move_tuple):
    if move_tuple:
        return 1000 if isThereWinMove([move_tuple]) else 0
    else:
        return 0


def isWinOrLoseMove(board, move_tuple):
    assert isinstance(board, Kamisado)
    win_res = isWinMove(move_tuple)
    lose_res = isLostMove(board, move_tuple)
    return win_res + lose_res


random_player = RandomAgent()
tower_progress_agent = TowerProgressAgent(0)
striking_position_agent = StrikingPositionAgent(0)
possible_moves_agent = PossibleMovesAgent(0)


def get_gp_play_move(gp_policy):
    def gp_play_move(board):
        assert isinstance(board, Kamisado)
        moves_ranks = Counter({move_tuple: gp_policy(board, move_tuple) for move_tuple in getPossibleMoves(board)})
        selected_move = moves_ranks.most_common(1)[0][0]
        return selected_move

    return gp_play_move


def kamisado_simulator(p1_play_move, p2_play_move, max_steps_num=10000, init_board=None):
    board = Kamisado(init_board=init_board)
    players = [p1_play_move, p2_play_move]
    none_count = 0
    i = 0
    # while not board.is_game_won() and none_count < 10:
    for i in range(max_steps_num):
        play_move = players[i % len(players)]
        move_tuple = play_move(board)
        move_tuple = move_tuple if move_tuple != () else random_player.play(board)
        tower, move = move_tuple
        board = board.move_tower(tower, move)
        none_count = none_count + 1 if not move else 0
        if board.is_game_won() or none_count >= 3:
            break
    return board, i


def evalSolver(individual, games=50):
    start = timeit.default_timer()
    gp_policy = toolbox.compile(expr=individual)

    ea_play_move = get_gp_play_move(gp_policy)
    # print(individual)
    games_won = 0
    games_lost = 0
    games_tie = 0
    moves_counts = []
    tower_progress_list = []
    striking_position_list = []
    possible_moves_list = []
    max_steps_num = 100
    # for i in range(5):
    #     p2_play = random_player.play
    #     board, moves_count = kamisado_simulator(ea_play_move, p2_play, max_steps_num)
    #     games_lost, games_tie, games_won1 = get_stats(board, max_steps_num, moves_count, moves_counts,
    #                                                   possible_moves_list, striking_position_list, tower_progress_list,
    #                                                   Player.WHITE)
    #
    #     board, moves_count = kamisado_simulator(p2_play, ea_play_move, max_steps_num)
    #     res = board.is_game_won()
    #     games_lost, games_tie, games_won2 = get_stats(board, max_steps_num, moves_count, moves_counts,
    #                                                   possible_moves_list, striking_position_list, tower_progress_list,
    #                                                   Player.BLACK)
    #     games_won += games_won1 + games_won2

    for board_init in [None, [3, 5, 2, 6, 1, 7, 0, 4], [0, 2, 4, 6, 1, 3, 5, 7]]:
        tower_progress_agent = TowerProgressAgent(0)
        striking_position_agent = StrikingPositionAgent(0)
        possible_moves_agent = PossibleMovesAgent(0)
        for agent in [random_player, tower_progress_agent, striking_position_agent]:
            # for agent in [tower_progress_agent]:
            p2_play = agent.play
            board, moves_count = kamisado_simulator(ea_play_move, p2_play, max_steps_num, init_board=board_init)
            games_lost, games_tie, games_won1 = get_stats(board, max_steps_num, moves_count, moves_counts,
                                                          possible_moves_list, striking_position_list,
                                                          tower_progress_list,
                                                          Player.WHITE)

            board, moves_count = kamisado_simulator(p2_play, ea_play_move, max_steps_num, init_board=board_init)
            res = board.is_game_won()
            games_lost, games_tie, games_won2 = get_stats(board, max_steps_num, moves_count, moves_counts,
                                                          possible_moves_list, striking_position_list,
                                                          tower_progress_list,
                                                          Player.BLACK)
            games_won += games_won1 + games_won2

    end = timeit.default_timer()
    # print(f'time {end - start} sec')
    tree_length = len(individual)
    # return games_won, np.mean(moves_counts), np.mean(tower_progress_list), np.mean(striking_position_list), np.mean(
    #     possible_moves_list)
    return games_won, np.mean(moves_counts)


def get_stats(board, max_steps_num, moves_count, moves_counts, possible_moves_list, striking_position_list,
              tower_progress_list, max_player=Player.WHITE):
    games_won = 0
    games_lost = 0
    games_tie = 0
    res = board.is_game_won()
    if res == max_player:
        games_won += 1
        moves_counts.append(2 - moves_count / max_steps_num)
        tower_progress_list.append(2 + tower_progress_agent.evaluate_game(board, max_player))
        striking_position_list.append(2 + striking_position_agent.evaluate_game(board, max_player))
        possible_moves_list.append(2 + possible_moves_agent.evaluate_game(board, max_player))

    elif res is None:
        games_tie += 1
        moves_counts.append(moves_count / max_steps_num)
        tower_progress_list.append(tower_progress_agent.evaluate_game(board, max_player))
        striking_position_list.append(striking_position_agent.evaluate_game(board, max_player))
        possible_moves_list.append(possible_moves_agent.evaluate_game(board, max_player))
    else:
        games_lost += 1
        moves_counts.append(moves_count / max_steps_num)
        tower_progress_list.append(tower_progress_agent.evaluate_game(board, max_player))
        striking_position_list.append(striking_position_agent.evaluate_game(board, max_player))
        possible_moves_list.append(possible_moves_agent.evaluate_game(board, max_player))
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
# pset.addPrimitive(getBoard, [Kamisado], Kamisado)
# pset.addPrimitive(getCurrentPlayer, [Kamisado], Player)
# pset.addPrimitive(getOtherPlayer, [Kamisado], Player)
# pset.addPrimitive(tower_progress_eval, [Kamisado, Player], float)
# pset.addPrimitive(striking_position_eval, [Kamisado, Player], float)
# pset.addPrimitive(possible_position_eval, [Kamisado, Player], float)
pset.addPrimitive(moveTower, [Kamisado, tuple], Kamisado)
pset.addPrimitive(getTotalTowerProgress, [Kamisado], float)
pset.addPrimitive(getMoveTowerProgress, [Kamisado, tuple], float)
pset.addPrimitive(moveSrikingPositionCount, [Kamisado, tuple], float)
pset.addPrimitive(getPossibleMovesCount, [Kamisado], float)
pset.addPrimitive(getEnemyPossibleMovesCount, [Kamisado, tuple], float)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(isWinOrLoseMove, [Kamisado, tuple], float)
# pset.addPrimitive(isLostMove, [Kamisado, tuple], float)
# pset.addPrimitive(isWinMove, [tuple], float)
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
pset.addTerminal(Player.WHITE.value, Player)
pset.addTerminal(Player.BLACK.value, Player)
pset.addTerminal([], list)
# pset.addTerminal((), tuple)
for i in range(11):
    pset.addTerminal(i / 10, float)
    pset.addTerminal(i, float)
pset.addTerminal(100, float)
pset.addTerminal(-100, float)
pset.renameArguments(ARG0="Board")
pset.renameArguments(ARG1="move_tuple")
# pset.addTerminal(Kamisado(), Kamisado)

# creator.create("FitnessMax", base.Fitness, weights=(10.0, 0.5, 1.0, 1.0, 1.0))
creator.create("FitnessMax", base.Fitness, weights=(10.0, 0.5))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

max_tree_length = 30
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSolver)
# toolbox.register("select", selAgentTournament, tournsize=5)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=7)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_length))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_length))

games_won = tools.Statistics(lambda ind: ind.fitness.values[0])
# [tower_progress_sum, striking_position_sum, possible_moves_sum]
move_count_mean = tools.Statistics(lambda ind: ind.fitness.values[1])
# progress_mean = tools.Statistics(lambda ind: ind.fitness.values[2])
# strike_mean = tools.Statistics(lambda ind: ind.fitness.values[3])
# possible_moves_mean = tools.Statistics(lambda ind: ind.fitness.values[4])
# mstats = tools.MultiStatistics(games_won=games_won, move_count_mean=move_count_mean, progress_mean=progress_mean,
#                                strike_mean=strike_mean, possible_moves_mean=possible_moves_mean)
mstats = tools.MultiStatistics(games_won=games_won, move_count_mean=move_count_mean)
# mstats = wins_stats
mstats.register("Avg", np.mean)
mstats.register("Std", np.std)
# mstats.register("Median", np.median)
mstats.register("Min", np.min)
mstats.register("Max", np.max)

pop_size = 100
pop = toolbox.population(n=pop_size)
hof = tools.HallOfFame(1)

games_count = 5
cxpb = 0.7
mutpb = 0.02
ngen = 100
print(f'pop{pop_size}_gen{ngen}_cxpb{cxpb}_mutpb{mutpb}_max{max_tree_length}')
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats=mstats,
                                   halloffame=hof, verbose=True)
print(hof[0])

output_path = 'data/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# [popolaation_size, cx_p, mut_p, gen, max_tree_length,seed]

# tower_progress_agent = TowerProgressAgent()
# striking_position_agent = StrikingPositionAgent()
# possible_moves_agent = PossibleMovesAgent()

p1_move = get_playe_move_from_policy(hof[0])
print(evalSolver(hof[0]))

cols = ['p1', 'p2', 'win', 'tie', 'lose']
train_rows = []
print('############################Train Data###################################')

for board_init in [None, [3, 5, 2, 6, 1, 7, 0, 4], [0, 2, 4, 6, 1, 3, 5, 7], [1, 3, 5, 7, 0, 2, 4, 6]]:
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
for j in range(10):
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
