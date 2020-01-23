import random
import operator
import math

import numpy

from deap import base
from deap import creator
from deap import gp
from deap import tools

# Define new functions
from KamisadoGame.kamisado import Kamisado
from gp_policy_agent import *

best_individual = None


def evalSolver(individual, games=50):
    start = timeit.default_timer()
    gp_policy = toolbox.compile(expr=individual)

    ea_play_move = get_gp_play_move(gp_policy)
    # ea_play_move = random_player.play
    # print(individual)
    games_won = 0
    moves_counts = []
    tower_progress_list = []
    striking_position_list = []
    possible_moves_list = []
    possible_striking_list = []
    max_steps_num = 100

    list(random.sample(range(8), 8))
    play_boards = [None, [3, 5, 2, 6, 1, 7, 0, 4], [0, 2, 4, 6, 1, 3, 5, 7]]
    depths = [0, 1, 2]
    p = [0.9, 0.07, 0.03]
    tower_progress_agent = TowerProgressAgent(np.random.choice(depths, 1, p=p))
    striking_position_agent = StrikingPositionAgent(np.random.choice(depths, 1, p=p))
    possible_moves_agent = PossibleMovesAgent(np.random.choice(depths, 1, p=p))
    # possible_striking_agent = PossibleStrikingAgent(0)
    # opponent = random.choice([possible_moves_agent, tower_progress_agent, striking_position_agent])
    opponents = [possible_moves_agent.play, tower_progress_agent.play, striking_position_agent.play]
    if best_individual:
        best_gp_policy = toolbox.compile(expr=best_individual)
        best_ea_play_move = get_gp_play_move(best_gp_policy)
        opponents.append(best_ea_play_move)

    for board_init in play_boards:
        for p2_play in opponents:
            # p2_play = agent.play
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
        # print(f'minmax player run time {sum(possible_moves_agent._play_run_times)} avg play {np.mean(possible_moves_agent._play_run_times)}')

    end = timeit.default_timer()
    # print(f'time {end - start} sec')
    tree_length = len(individual)
    return 1 + (games_won / (len(play_boards) * 4)), np.mean(moves_counts)
    # return games_won, np.mean(moves_counts), np.mean(tower_progress_list), np.mean(striking_position_list)
    # return games_won, np.mean(moves_counts)


def get_stats(board, max_steps_num, moves_count, moves_counts, possible_moves_list, striking_position_list,
              tower_progress_list, possible_striking_list, max_player=Player.WHITE):
    games_won = 0
    games_lost = 0
    games_tie = 0
    res = board.is_game_won()
    if res == max_player:
        games_won += 1
        # moves_counts.append(2)
        moves_counts.append(2.30 - moves_count / max_steps_num)
        # tower_progress_list.append(towerProgressEval(board, max_player))
        # striking_position_list.append(strikingPossitionEval(board, max_player))
        # possible_moves_list.append(possibleMovesEval(board, max_player))
        # possible_striking_list.append(25)

    elif res is None:
        games_tie += 1
        moves_counts.append(1 + moves_count / max_steps_num)
        # tower_progress_list.append(towerProgressEval(board, max_player))
        # striking_position_list.append(strikingPossitionEval(board, max_player))
        # possible_moves_list.append(possibleMovesEval(board, max_player))
        # possible_striking_list.append(possible_striking_agent.evaluate_game(board, max_player))
    else:
        games_lost += 1
        moves_counts.append(1 + moves_count / max_steps_num)
    # tower_progress_list.append(towerProgressEval(board, max_player))
    # striking_position_list.append(strikingPossitionEval(board, max_player))
    # possible_moves_list.append(possibleMovesEval(board, max_player))
    return games_lost, games_tie, games_won


def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    global best_individual
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=operator.attrgetter(fit_attr)))
    best_individual = max(individuals, key=operator.attrgetter(fit_attr))
    return chosen


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def get_playe_move_from_policy(p1):
    gp_policy = toolbox.compile(expr=p1)
    p1_move = get_gp_play_move(gp_policy)
    return p1_move


adfset2 = gp.PrimitiveSetTyped("ADF2", [Kamisado, tuple], float)
adfset2.addPrimitive(operator.add, [float, float], float)
adfset2.addPrimitive(operator.sub, [float, float], float)
adfset2.addPrimitive(operator.mul, [float, float], float)
adfset2.addPrimitive(protectedDiv, [float, float], float)
adfset2.addPrimitive(operator.neg, [float], float)
adfset2.addPrimitive(getMove, [tuple], tuple)
adfset2.addPrimitive(moveTower, [Kamisado, tuple], Kamisado)
adfset2.addPrimitive(getStrikingPositionFrec, [Kamisado], float)
adfset2.addPrimitive(getOpenEndPostionsCount, [Kamisado], float)
adfset2.addPrimitive(moveSrikingPositionCount, [Kamisado, tuple], float)
adfset2.addPrimitive(isWinOrLoseMove, [Kamisado, tuple], float)
adfset2.addEphemeralConstant("rand2", lambda: random.randint(-10, 10), float)

adfset1 = gp.PrimitiveSetTyped("ADF1", [Kamisado, tuple], float)
adfset1.addPrimitive(operator.add, [float, float], float)
adfset1.addPrimitive(operator.sub, [float, float], float)
adfset1.addPrimitive(operator.mul, [float, float], float)
adfset1.addPrimitive(protectedDiv, [float, float], float)
adfset1.addPrimitive(operator.neg, [float], float)
# adfset1.addADF(adfset2)
adfset1.addPrimitive(getMove, [tuple], tuple)
adfset1.addPrimitive(moveTower, [Kamisado, tuple], Kamisado)
adfset1.addPrimitive(getTowerPassHafe, [Kamisado], float)
adfset1.addPrimitive(getTowerProgressFrec, [Kamisado], float)
adfset1.addPrimitive(getMoveDistanceFromStart, [Kamisado, tuple], float)
adfset1.addPrimitive(isWinOrLoseMove, [Kamisado, tuple], float)
adfset1.addEphemeralConstant("rand1", lambda: random.randint(-10, 10), float)

adfset0 = gp.PrimitiveSetTyped("ADF0", [Kamisado, tuple], float)
adfset0.addPrimitive(operator.add, [float, float], float)
adfset0.addPrimitive(operator.sub, [float, float], float)
adfset0.addPrimitive(operator.mul, [float, float], float)
adfset0.addPrimitive(protectedDiv, [float, float], float)
adfset0.addPrimitive(operator.neg, [float], float)
# adfset0.addADF(adfset1)
# adfset0.addADF(adfset2)
adfset0.addPrimitive(getMove, [tuple], tuple)
adfset0.addPrimitive(moveTower, [Kamisado, tuple], Kamisado)
adfset0.addPrimitive(getPossiblePossitionFrec, [Kamisado], float)
adfset0.addPrimitive(getPossibleMovesCount, [Kamisado], float)
adfset0.addPrimitive(getEnemyPossibleMovesCount, [Kamisado, tuple], float)
adfset0.addPrimitive(isWinOrLoseMove, [Kamisado, tuple], float)
adfset0.addEphemeralConstant("rand0", lambda: random.randint(-10, 10), float)

pset = gp.PrimitiveSetTyped("MAIN", [Kamisado, tuple], float)
pset.addEphemeralConstant("rand101", lambda: random.randint(-10, 10), float)
pset.addADF(adfset0)
pset.addADF(adfset1)
pset.addADF(adfset2)
pset.addPrimitive(getMove, [tuple], tuple)
pset.addPrimitive(moveTower, [Kamisado, tuple], Kamisado)
pset.addPrimitive(isWinOrLoseMove, [Kamisado, tuple], float)
pset.addPrimitive(getPossiblePossitionFrec, [Kamisado], float)
pset.addPrimitive(getPossibleMovesCount, [Kamisado], float)
pset.addPrimitive(getEnemyPossibleMovesCount, [Kamisado, tuple], float)
pset.addPrimitive(getTowerPassHafe, [Kamisado], float)
pset.addPrimitive(getTowerProgressFrec, [Kamisado], float)
pset.addPrimitive(getMoveDistanceFromStart, [Kamisado, tuple], float)
pset.addPrimitive(isWinOrLoseMove, [Kamisado, tuple], float)
pset.addPrimitive(getStrikingPositionFrec, [Kamisado], float)
pset.addPrimitive(moveSrikingPositionCount, [Kamisado, tuple], float)
pset.addPrimitive(getOpenEndPostionsCount, [Kamisado], float)
pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)
pset.addEphemeralConstant("rand3", lambda: random.randint(-10, 10), float)
pset.renameArguments(ARG0="Board")
pset.renameArguments(ARG1="move_tuple")

psets = (pset, adfset0, adfset1, adfset2)

creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
creator.create("Tree", gp.PrimitiveTree)

creator.create("Individual", list, fitness=creator.FitnessMin)

max_tree_length = 8
toolbox = base.Toolbox()
toolbox.register('adf_expr0', gp.genFull, pset=adfset0, min_=2, max_=4)
toolbox.register('adf_expr1', gp.genFull, pset=adfset1, min_=2, max_=4)
toolbox.register('adf_expr2', gp.genFull, pset=adfset2, min_=2, max_=4)
toolbox.register('main_expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=max_tree_length)

toolbox.register('ADF0', tools.initIterate, creator.Tree, toolbox.adf_expr0)
toolbox.register('ADF1', tools.initIterate, creator.Tree, toolbox.adf_expr1)
toolbox.register('ADF2', tools.initIterate, creator.Tree, toolbox.adf_expr2)
toolbox.register('MAIN', tools.initIterate, creator.Tree, toolbox.main_expr)

func_cycle = [toolbox.MAIN, toolbox.ADF0, toolbox.ADF1, toolbox.ADF2]

toolbox.register('individual', tools.initCycle, creator.Individual, func_cycle)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(individual)
    # Evaluate the sum of squared difference between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    values = (x / 10. for x in range(-10, 10))
    diff_func = lambda x: (func(x) - (x ** 4 + x ** 3 + x ** 2 + x)) ** 2
    diff = sum(map(diff_func, values))
    return diff,


toolbox.register('compile', gp.compileADF, psets=psets)
toolbox.register('evaluate', evalSolver)
toolbox.register('select', selTournament, tournsize=3)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('expr', gp.genFull, min_=1, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr)


def main():
    random.seed(1024)
    ind = toolbox.individual()

    pop_size = 50
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    games_won = tools.Statistics(lambda ind: ind.fitness.values[0])
    move_count_mean = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats = tools.MultiStatistics(games_won=games_won, move_count_mean=move_count_mean)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    # logbook.header = "gen", "evals", "std", "min", "avg", "max"

    games_count = 1
    cxpb, mutpb, ngen = 0.5, 0.01, 100
    experiment_name = f'pop{pop_size}_gen{ngen}_cxpb{cxpb}_mutpb{mutpb}_max{max_tree_length}_adf'

    # Evaluate the entire population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    for g in range(1, ngen):
        # Select the offspring
        offspring = toolbox.select(pop, len(pop))
        # Clone the offspring
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            for tree1, tree2 in zip(ind1, ind2):
                if random.random() < cxpb:
                    toolbox.mate(tree1, tree2)
                    del ind1.fitness.values
                    del ind2.fitness.values

        for ind in offspring:
            for tree, pset in zip(ind, psets):
                if random.random() < mutpb:
                    toolbox.mutate(individual=tree, pset=pset)
                    del ind.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalids:
            ind.fitness.values = toolbox.evaluate(ind)

        # Replacement of the population by the offspring
        pop = offspring
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalids), **record)
        print(logbook.stream)

    print('Best individual : ', hof[0][0], hof[0].fitness)

    print(hof[0])

    output_path = 'data/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    import matplotlib.pyplot as plt

    stats_names = ['fitness']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40, 8))
    for idx, statistics in enumerate(
            stats_names):
        gen = logbook.select("gen")
        fit_mins = logbook.select("Min")
        fit_avgs = logbook.select("Avg")
        fit_maxs = logbook.select("Max")
        fit_medians = logbook.select("Median")
        ax.plot(gen, fit_mins, "b-", label="Minimum Fitness")
        ax.plot(gen, fit_avgs, "r-", label="Average Fitness")
        ax.plot(gen, fit_maxs, "g-", label="Max Fitness")
        ax.set_xlabel("Generation", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.set_ylabel(f"{statistics}", color="b", fontsize=18)
        ax.legend(loc="lower right", fontsize=14)
        ax.set_title(f'Kamisado agents performance on {statistics}', fontsize=18)

    try:
        plt.savefig(f'{experiment_name}.png', dpi=fig.dpi)
    except Exception as e:
        print(e)
    plt.show()

    # exit(1)
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
    file_name = f'train_results_pop{pop_size}_gen{ngen}_cxpb{cxpb}_mutpb{mutpb}_max{max_tree_length}_ADF.csv'
    train_df.to_csv(os.path.join(output_path, file_name))

    test_rows = []
    print('############################Test Data###################################')
    for j in range(3):
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
    file_name = f'test_results_pop{pop_size}_gen{ngen}_cxpb{cxpb}_mutpb{mutpb}_max{max_tree_length}_ADF.csv'
    test_df.to_csv(os.path.join(output_path, file_name))

    return pop, stats, hof


if __name__ == "__main__":
    main()
