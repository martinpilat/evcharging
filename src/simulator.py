import pandas as pd
import numpy as np
import datetime
import planners
import heapq

import utils
import settings

np.seterr(all='raise')

class ActiveRequest:

    def __init__(self, available_after, available_until, remaining_charge, max_charging_speed, houseid, initial_steps, initial_charge):
        self.available_until = available_until
        self.remaining_charge = remaining_charge
        self.max_charging_speed = max_charging_speed
        self.houseid = houseid
        self.available_after = available_after
        self.initial_steps = initial_steps
        self.initial_charge = initial_charge

class Simulator:

    def __init__(self, requests_file, data_file, baseline_file, baseline_weight, start_time, planner):
        self.baseline_weight = baseline_weight
        self.requests_frame = pd.read_csv(requests_file, parse_dates=[4,5]).sort_values(by='available_after')
        self.baseline_consumption = pd.read_csv(baseline_file, parse_dates=[0], header=None)
        self.baseline_consumption.columns = ['datetime', 'consumption']
        self.baseline_consumption.set_index('datetime', inplace=True)
        self.baseline_consumption = self.baseline_consumption['consumption']
        self.electricity_use = pd.read_csv(data_file, parse_dates=[0])
        self.electricity_use.set_index('datetime', inplace=True)
        self.electricity_use.fillna(method='ffill', inplace=True)
        self.overall_consumption = self.electricity_use.sum(axis=1) + self.baseline_weight*self.baseline_consumption
        self.requests = []

        if planner:
            self.reset(start_time, planner)

    def reset(self, start_time, planner):
        self.current_time = start_time
        self.requests = []
        self.active_requests = []
        heapq.heapify(self.active_requests)
        self.active_households = set([])
        self.missed_charge = 0
        self.last_total_charge = 0
        self.planner = planner

        self.planners = {houseid: self.planner.copy() for houseid in np.unique(self.requests_frame.houseid)}

        for req in self.requests_frame.itertuples():

            r = ActiveRequest(available_after=req.available_after,
                              available_until=req.available_before,
                              remaining_charge=req.required_charge,
                              max_charging_speed=req.max_charging_speed,
                              houseid=req.houseid,
                              initial_steps=(req.available_before - req.available_after) // settings.TIME_STEP,
                              initial_charge=req.required_charge)

            if not np.isnan(r.remaining_charge) and r.remaining_charge > 0.01:
                self.requests.append(r)

        self.requests = list(sorted(self.requests, key=lambda r: r.available_after))

        self.req_idx = 0
        while self.requests[self.req_idx].available_after <= start_time - settings.TIME_STEP:
            self.req_idx += 1

        self.simulation_log = []

    def step(self):

        end = self.current_time + settings.TIME_STEP

        # remove cars that left in the last time step
        while self.active_requests and self.active_requests[0][0] < end:
            self.missed_charge += self.active_requests[0][2].remaining_charge # compute missed charge
            _, r_idx, r = heapq.heappop(self.active_requests)
            if r.houseid in self.active_households:
                self.active_households.remove(r.houseid)
            else:
                print(r.houseid, r_idx, r.available_after, r.available_until, r.remaining_charge, r.initial_charge)
                raise(KeyError(r.houseid))

        while self.req_idx < len(self.requests) and self.requests[self.req_idx].available_after <= self.current_time:
            r = self.requests[self.req_idx]
            if r.houseid not in self.active_households:
                # in rare cases there are overlapping requests, skip them, probably an artifact of the way
                # how requests are generated

                heapq.heappush(self.active_requests, (r.available_until, self.req_idx, r))
                self.active_households.add(r.houseid)

            self.req_idx += 1

        # charge cars
        total_charge = 0
        overall_consumption = self.overall_consumption.at[self.current_time]

        for house_id, planner in self.planners.items():
            # give each planner info on last consumption of household and last overall consumption
            if house_id not in self.active_households: # update household without active requests, other will be updated later
                planner.update_info(self.electricity_use.at[self.current_time, house_id],
                                    overall_consumption + self.last_total_charge, current_time=self.current_time)

        for _, _, ar in self.active_requests:
            self.planners[ar.houseid].update_info(self.electricity_use.loc[self.current_time, ar.houseid],
                                                  overall_consumption + self.last_total_charge,
                                                  ar=ar, current_time=self.current_time)
            if ar.remaining_charge < 0.0001:
                continue  # already charged, skip request
            remaining_steps = utils.remaining_steps(ar, self.current_time)
            min_charging_speed = utils.minimum_charging_speed(remaining_steps, ar)
            raw_charge = self.planners[ar.houseid].get_charge(remaining_steps, ar, current_time=self.current_time)
            charge = min(ar.max_charging_speed, raw_charge) # ensure the charging is not faster than max
            charge = max(min_charging_speed, charge) # make sure the charge is able to charge the vehicle
            charge = min(charge, ar.remaining_charge*2) # make sure we do not over-charge the vehicle
            ar.remaining_charge -= charge/2
            total_charge += charge

        self.last_total_charge = total_charge

        self.simulation_log.append((self.current_time, total_charge, overall_consumption))

        self.current_time += settings.TIME_STEP

    # def get_gradient_data(self):
    #    return [p.last_states for _, p in self.planners.items()], self.simulation_log

def test_planner(sim, model, log_prefix='test', run_id=-1, start_date=datetime.datetime(2013, 3, 1, 0, 0), 
                 end_date=datetime.datetime(2013, 4, 1, 0, 0)):

    sim.reset(start_time=start_date, planner=planner)

    while sim.current_time < end_date:
        sim.step()

    loads = [tc + oc for (_, tc, oc) in sim.simulation_log][2 * 24:]
    times = [t for (t, _, _) in sim.simulation_log][2 * 24:]
    baseline_load = [oc for (_, _, oc) in sim.simulation_log][2 * 24:]

    results = {
        'objective': np.std(loads),
        'max_load': np.max(loads),
        'min_load': np.min(loads),
        'perc25': np.percentile(loads, 2.5),
        'perc975': np.percentile(loads, 97.5),
    }

    if not os.path.exists('../results/plots'):
        os.makedirs('../results/plots')

    json.dump(results, open(f'../results/{log_prefix}_results_{run_id}.json', 'w'), indent=1)
    pickle.dump((times, loads, baseline_load), open(f'../results/plots/{log_prefix}_plotdata_{run_id}.pkl', 'wb'))

    start = 48
    end = 48 + 168 * 2

    import matplotlib.pyplot as plt

    plt.figure(figsize=(4.8, 4))
    plt.xticks(rotation='vertical')
    plt.step(times[start:end], loads[start:end], label='planner')
    plt.step(times[start:end], baseline_load[start:end], label='baseline', linestyle='--', linewidth=1, color='black')    
    plt.xlim(datetime.datetime(2013, 3, 3), datetime.datetime(2013, 3, 7))
    plt.legend(ncol=3, loc='upper center')
    plt.xlabel('Time')
    plt.ylabel('Load [kW]')
    plt.tight_layout()

    plt.savefig(f'../results/plots/{log_prefix}_plot_{run_id}.pdf')

    return results

if __name__ == '__main__':

    import numpy as np
    import json
    import pickle
    import os

    oc = pd.read_csv('../data/optimum_charge.csv', parse_dates=[0], index_col=['datetime'])

    planner=planners.OptimumPlanner(optimum_charges=oc)
    
    sim = Simulator(requests_file='../data/requests_1.csv', data_file='../data/selected_1.csv',
                    baseline_file='../data/baseline_1.csv', baseline_weight=50/450,
                    start_time=datetime.datetime(2013, 3, 1, 0, 0), 
                    planner=None)

    # test_planner(sim, planner, start_date=datetime.datetime(2013, 1, 1), end_date=datetime.datetime(2013,2,1))

    print(test_planner(sim, planner, log_prefix='quad_prog_opt', run_id=0))

    # for k, v in sim.planners.items():
    #     v.write_training_data(k)

    # for i in range(10):
    #     print('='*20 + f' Run {i+1} ' + '='*20)
    #     planner=planners.SavedPlanner(f'../weights/weights_stable0.1_hh_50_rs_{i}.npy', [15, 50, 1], activations=[utils.relu, utils.sigmoid], hh_only=True)
    #     print(test_planner(sim, planner, log_prefix='time_stable0.1_50_rs', run_id=i))