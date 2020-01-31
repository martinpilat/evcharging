import argparse
import datetime
import functools
import json
import multiprocessing
import os
import pickle

import numpy

import evolution
import planners
import simulator
import utils

sim = simulator.Simulator(requests_file='../data/requests_1.csv',
                          data_file='../data/selected_1.csv',
                          baseline_file='../data/baseline_1.csv',
                          baseline_weight=50/450,
                          start_time=datetime.datetime(2013, 1, 3, 16, 0), planner=planners.RandomPlanner())


def grad_helper(x):
    return (x[2](x[0], start=x[4], steps=2 * 24 + 2 * 2 * 24) - x[3]) / x[1]

def gradient(f, x, fx=None, start=None, pool=None):
    fx = f(x, start=start, steps=2*24+2*2*24)
    xphs = []
    for i in range(len(x)):
        xph = numpy.copy(x)
        xph[i] = xph[i] + 0.000000001
        d = xph[i] - x[i]
        xphs.append((xph, d, f, fx, start))

    if pool:
        grad = pool.map(grad_helper, xphs)
    else:
        grad = map(grad_helper, xphs)

    return numpy.array(grad)

def func_2(x, config, additional_planner_params, start=None, steps=None):
    return evolution.evaluate_fitness(x, config = config, additional_planner_params = additional_planner_params,
                                      start=start, steps=steps)[0]


if __name__ == '__main__':

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Evolution of EV charging controllers')
    parser.add_argument('-c', '--config', help='configuration file', type=str, required=True)
    parser.add_argument('-s', '--seed', help='seed of random number generator', type=int, default=-1)

    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))

    pool = multiprocessing.Pool(config['cpus'])

    seed = args.seed
    if seed == -1:
        seed = numpy.random.randint(0, 1_000_000_000)

    log_prefix = config['log_prefix']

    numpy.random.seed(seed)

    planner = eval(config['planner'])
    additional_params = None
    if isinstance(planner, planners.ESNPlanner):
        additional_params = {'W_in': planner.esn.W_in, 'W': planner.esn.W}
    N = planner.vectorized_size()

    func = functools.partial(func_2, config=config, additional_planner_params=additional_params)

    x = config['sigma']*numpy.random.randn(N)
    #x = numpy.load(open('../results/advnn_all_best_sol_13767.npy', 'rb'))
    best_sol = x.copy()
    fx = func(x)
    bf = fx

    for i in range(config['max_gen']):

        print(i, bf)
        startday = numpy.random.randint(3, 14)
        starthour = numpy.random.randint(0, 24)
        g = gradient(func, x, fx=None, start=datetime.datetime(2013, 1, startday, starthour, 0), pool=pool)
        max_g = numpy.max(numpy.abs(g))
        if max_g > 100:
            print(f'Warning: gradient to large ({max_g}), skipping batch')
            continue
        print(max_g)
        aa = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        fs = pool.map(func, [(x - a*g) for a in aa])
        print(fs)
        ba = numpy.argmin(fs)
        x = x.copy() - aa[ba]*g
        fx = fs[ba]
        if fx <= bf:
            bf = fx
            best_sol = x.copy()

    planner = eval(config['planner'])
    planner.set_network(best_sol)
    if isinstance(planner, planners.ESNPlanner):
        planner.set_additional_params(additional_params)

    numpy.save(f'../results/{log_prefix}_best_sol_{seed}.npy', best_sol)
    if isinstance(planner, planners.ESNPlanner):
        pickle.dump((planner.esn.W_in, planner.esn.W, planner.esn.alpha),
                    open(f'../results/{log_prefix}_best_sol_additional_{seed}.pkl', 'wb'))

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

    start = 24 * 2
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