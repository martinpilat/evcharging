import argparse
import datetime
from deap import base, algorithms, cma, creator, tools
import json
import multiprocessing
import numpy
import os
import pickle

import simulator
import planners
import utils

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

sim = simulator.Simulator(requests_file='../data/requests_1.csv',
                          data_file='../data/selected_1.csv',
                          baseline_file='../data/baseline_1.csv',
                          baseline_weight=50/450,
                          start_time=datetime.datetime(2013, 1, 3, 16, 0), planner=planners.RandomPlanner())

def evaluate_fitness(ind, config, additional_planner_params=None, start=None, steps=None):
    print('.', end='', flush=True)
    planner = eval(config['planner'])
    planner.set_network(ind)
    if additional_planner_params:
        planner.set_additional_params(additional_planner_params)

    if not start:
        start=datetime.datetime(2013, 1, 3, 16)

    if not steps:
        steps = 2*168*2+2*24

    sim.reset(start_time=start, planner=planner)
    for i in range(steps):
        sim.step()

    # max_tc_oc = max(tc+oc for (_, tc, oc) in sim.simulation_log)
    # min_tc_oc = min(tc+oc for (_, tc, oc) in sim.simulation_log)
    #
    # max_oc = max(oc for (_, _, oc) in sim.simulation_log)
    # min_oc = min(oc for (_, _, oc) in sim.simulation_log)

    # fitness = numpy.sqrt(numpy.mean([(tc+oc)**2 for (_, tc, oc) in sim.simulation_log[4*24:]]))  # ignore first day to pre-warm the model
    fitness = numpy.std([(tc + oc) for (_, tc, oc) in sim.simulation_log[2 * 24:]])  # ignore first day to pre-warm the model
    # print(f"fitness: {fitness}")

    #gdata = sim.get_gradient_data()
    #print(gdata)

    return fitness,


def main():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Evolution of EV charging controllers')
    parser.add_argument('-c', '--config', help='configuration file', type=str, required=True)
    parser.add_argument('-d', '--de', help='use differential evolution', action="store_true")
    parser.add_argument('-s', '--seed', help='seed of random number generator', type=int, default=-1)

    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))

    seed = args.seed
    if seed == -1:
        seed = numpy.random.randint(0, 1_000_000_000)

    log_prefix = config['log_prefix']

    numpy.random.seed(seed)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    toolbox = base.Toolbox()

    pool = multiprocessing.Pool(config['cpus'])
    toolbox.register('map', pool.map)

    planner = eval(config['planner'])
    additional_params = None
    if isinstance(planner, planners.ESNPlanner):
        additional_params = {'W_in': planner.esn.W_in, 'W': planner.esn.W}
    N = planner.vectorized_size()

    toolbox.register("evaluate", evaluate_fitness, config=config, additional_planner_params=additional_params)

    # parent = creator.Individual(numpy.random.randn(N)*0.1)
    # parent.fitness.values = toolbox.evaluate(parent)
    # strategy = cma.StrategyOnePlusLambda(parent, sigma=0.1, lambda_=32)


    if args.de:

        print("Running DE...")
        toolbox.register('mutate', mutDE, f=0.8)
        toolbox.register("mate", cxExponential, cr=0.8)
        toolbox.register("select", tools.selRandom, k=3)
        toolbox.register("attr_float", rand_init)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, N)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, config['pop_size'])

        pop = toolbox.population()

        pop, log = differential_evolution(pop, toolbox, ngen=config['max_gen'], stats=stats, hof=hof)

    else:
        strategy = cma.Strategy(centroid=0.1 * numpy.random.randn(N), sigma=config['sigma'], lambda_=config['pop_size'])
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=config['max_gen'], stats=stats, halloffame=hof)


    planner = eval(config['planner'])
    planner.set_network(hof.items[0])
    if isinstance(planner, planners.ESNPlanner):
        planner.set_additional_params(additional_params)

    numpy.save(f'../results/{log_prefix}_best_sol_{seed}.npy', hof.items[0])
    if isinstance(planner, planners.ESNPlanner):
        pickle.dump((planner.esn.W_in, planner.esn.W, planner.esn.alpha),
                    open(f'../results/{log_prefix}_best_sol_additional_{seed}.pkl', 'wb'))
    pickle.dump(log, open(f'../results/{log_prefix}_stats_{seed}.pkl', 'wb'))
    pickle.dump(pop, open(f'../results/{log_prefix}_pop_{seed}.pkl', 'wb'))

    sim.reset(start_time=datetime.datetime(2013, 3, 1, 0, 0), planner=planner)

    while sim.current_time < datetime.datetime(2013, 4, 1, 0, 0):
        sim.step()

    loads = [tc + oc for (_, tc, oc) in sim.simulation_log][2 * 24:]
    times = [t for (t, _, _) in sim.simulation_log][2 * 24:]
    baseline_load = [oc for (_, _, oc) in sim.simulation_log][2 * 24:]

    results = {
        'objective': numpy.std(loads),
        'max_load': numpy.max(loads),
        'min_load': numpy.min(loads),
        'perc25': numpy.percentile(loads, 2.5),
        'perc975': numpy.percentile(loads, 97.5),
    }

    if not os.path.exists('../results/plots'):
        os.makedirs('../results/plots')

    json.dump(results, open(f'../results/{log_prefix}_results_{seed}.json', 'w'), indent=1)
    pickle.dump((times, loads, baseline_load), open(f'../results/plots/{log_prefix}_plotdata_{seed}.pkl', 'wb'))

    start = 48
    end = 48 + 168 * 2

    plt.figure(figsize=(4.8, 4))
    plt.xticks(rotation='vertical')
    plt.step(times[start:end], loads[start:end], label='planner')
    plt.step(times[start:end], baseline_load[start:end], label='baseline', linestyle='--', linewidth=1, color='black')    
    plt.xlim(datetime.datetime(2013, 3, 3), datetime.datetime(2013, 3, 7))
    plt.legend(ncol=3, loc='upper center')
    plt.xlabel('Time')
    plt.ylabel('Load [kW]')
    plt.tight_layout()

    plt.savefig(f'../results/plots/{log_prefix}_plot_{seed}.pdf')

def differential_evolution(pop, toolbox, ngen, stats, hof):

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = list(pop)
    # Evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    for g in range(1, ngen):
        children = []
        for agent in pop:
            # We must clone everything to ensure independance
            a, b, c = [toolbox.clone(ind) for ind in toolbox.select(pop)]
            x = toolbox.clone(agent)
            y = toolbox.clone(agent)
            y = toolbox.mutate(y, a, b, c)
            z = toolbox.mate(x, y)
            del z.fitness.values
            children.append(z)

        fitnesses = toolbox.map(toolbox.evaluate, children)
        for (i, ind), fit in zip(enumerate(children), fitnesses):
            ind.fitness.values = fit
            if ind.fitness > pop[i].fitness:
                pop[i] = ind

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        print(logbook.stream)

    return pop, logbook

def mutDE(y, a, b, c, f):
    for i in range(len(y)):
        y[i] = a[i] + f*(b[i]-c[i])
    return y

import itertools

def cxExponential(x, y, cr):
    size = len(x)
    index = numpy.random.randint(size)
    # Loop on the indices index -> end, then on 0 -> index
    for i in itertools.chain(range(index, size), range(0, index)):
        x[i] = y[i]
        if numpy.random.random() < cr:
            break
    return x

def rand_init():
    return 0.1*numpy.random.randn()

if __name__ == '__main__':
    main()

