import datetime
import numpy as np
import collections

import esn
import utils

class MinRequiredPlanner:

    def get_charge(self, remaining_steps, ar, **kwargs):
        return utils.minimum_charging_speed(remaining_steps, ar)

    def copy(self):
        return MinRequiredPlanner()

    def update_info(self, household_consumption, overall_consumption, **kwargs):
        pass

class RandomPlanner:

    def get_charge(self, remaining_steps, ar, **kwargs):
        return np.random.uniform(utils.minimum_charging_speed(remaining_steps, ar), ar.max_charging_speed)

    def copy(self):
        return RandomPlanner()

    def update_info(self, household_consumption, overall_consumption, **kwargs):
        pass

class MaxPossibleCharge:

    def get_charge(self, remaining_steps, ar, **kwargs):
        return ar.max_charging_speed

    def copy(self):
        return MaxPossibleCharge()

    def update_info(self, household_consumption, overall_consumption, **kwargs):
        pass

class ConstantCharge:

    def get_charge(self, remaining_steps, ar, **kwargs):
        return utils.constant_charging_speed(remaining_steps, ar)

    def copy(self):
        return ConstantCharge()

    def update_info(self, household_consumption, overall_consumption, **kwargs):
        pass

class NNPlanner:

    def __init__(self, layer_sizes, activations, hh_only=False):
        self.layer_sizes = layer_sizes
        self.layers = None
        self.hh_only = hh_only

        if isinstance(activations, collections.Iterable):
            if len(list(activations)) != len(layer_sizes) - 1:
                raise AttributeError("Number of activations does not match number of layers")
            self.activations = list(activations)
        else:
            self.activations = [activations]*(len(layer_sizes)-1)

        self.vectorized_net = None

        self.last_hh_consumptions = []
        self.last_overall_consumptions = []
        self.last_charging_consumptions = []
        self.last_charging = 0

    def copy(self):
        nn = NNPlanner(self.layer_sizes, self.activations, self.hh_only)
        nn.set_network(self.vectorized_net)
        return nn

    def vectorized_size(self):
        return sum(map(lambda x: (x[0]+1)*x[1] , zip(self.layer_sizes, self.layer_sizes[1:])))

    def set_network(self, vectorized_net):

        if len(vectorized_net) != self.vectorized_size():
            raise AttributeError(f"Length of vector does not match vectorized_size: {len(vectorized_net)} != {self.vectorized_size()}")

        self.vectorized_net = vectorized_net

        self.layers = []

        sum_sizes = 0
        for (p, n) in zip(self.layer_sizes, self.layer_sizes[1:]):
            layer = vectorized_net[sum_sizes: sum_sizes + (p+1)*n]
            self.layers.append(np.reshape(layer, newshape=(p+1, n)))
            sum_sizes += (p+1)*n

    def eval_network(self, inputs):

        activations = inputs
        try:
            for act_func, layer in zip(self.activations, self.layers):
                activations_1 = np.append(np.array([1.0]), activations) # add constant 1.0 for the bias term
                activations = act_func(np.dot(activations_1, layer))
        except Exception as e:
            print("Activations:", activations)
            raise e

        return activations

    def update_info(self, household_consumption, overall_consumption, **kwargs):
        if np.isnan(household_consumption):
            print(household_consumption)
        self.last_hh_consumptions.append(household_consumption + self.last_charging)
        self.last_overall_consumptions.append(overall_consumption)
        self.last_charging_consumptions.append(self.last_charging)

        self.last_charging = 0

        # store only last 24 hours
        self.last_charging_consumptions = self.last_charging_consumptions[-2 * 24:]
        self.last_overall_consumptions = self.last_overall_consumptions[-2 * 24:]
        self.last_hh_consumptions = self.last_hh_consumptions[-2 * 24:]

    def basic_inputs(self, ar, remaining_steps):

        try:
            max_recent_hh = np.max(self.last_hh_consumptions)
        except FloatingPointError:
            print(self.last_hh_consumptions)

        avg_recent_hh = np.mean(self.last_hh_consumptions)
        min_recent_hh = np.min(self.last_hh_consumptions)
        max_recent_overall = np.max(self.last_overall_consumptions)
        avg_recent_overall = np.mean(self.last_overall_consumptions)
        min_recent_overall = np.min(self.last_overall_consumptions)

        hh_consumption = 0
        overall_consumption = 0
        hh_change_last = 0
        overall_change_last = 0
        hh_change_1h = 0
        overall_change_1h = 0
        hh_change_3h = 0
        overall_change_3h = 0
        hh_ratio = 0
        overall_ratio = 0

        if avg_recent_hh > 0:
            hh_consumption = self.last_hh_consumptions[-1] / avg_recent_hh - 1
        if avg_recent_overall > 0:
            overall_consumption = self.last_overall_consumptions[-1] / avg_recent_overall - 1
        if max_recent_hh != min_recent_hh:
            hh_ratio = (self.last_hh_consumptions[-1] - min_recent_hh) / (max_recent_hh - min_recent_hh)
        if max_recent_overall != min_recent_overall:
            overall_ratio = (self.last_overall_consumptions[-1] - min_recent_overall) / (
                    max_recent_overall - min_recent_overall)
        if len(self.last_hh_consumptions) > 1 and self.last_hh_consumptions[-2] > 0:
            hh_change_last = self.last_hh_consumptions[-1] / self.last_hh_consumptions[-2] - 1
            overall_change_last = self.last_overall_consumptions[-1] / self.last_overall_consumptions[-2] - 1
        if len(self.last_hh_consumptions) > 2 and self.last_hh_consumptions[-3] > 0:
            hh_change_1h = self.last_hh_consumptions[-1] / self.last_hh_consumptions[-3] - 1
            overall_change_1h = self.last_overall_consumptions[-1] / self.last_overall_consumptions[-3] - 1
        if len(self.last_hh_consumptions) > 6 and self.last_hh_consumptions[-7] > 0:
            hh_change_3h = self.last_hh_consumptions[-1] / self.last_hh_consumptions[-7] - 1
            overall_change_3h = self.last_overall_consumptions[-1] / self.last_overall_consumptions[-7] - 1

        const_ch_speed = utils.constant_charging_speed(remaining_steps, ar) / ar.max_charging_speed
        min_ch_speed = utils.minimum_charging_speed(remaining_steps, ar) / ar.max_charging_speed

        if self.hh_only:
            inputs = np.array([ar.remaining_charge / ar.initial_charge, remaining_steps / ar.initial_steps,
                               const_ch_speed, min_ch_speed, hh_consumption, hh_change_last, hh_change_1h,
                               hh_change_3h, hh_ratio])
        else:
            inputs = np.array([ar.remaining_charge / ar.initial_charge, remaining_steps / ar.initial_steps,
                               const_ch_speed, min_ch_speed, hh_consumption, overall_consumption,
                               hh_change_last, overall_change_last, hh_change_1h, overall_change_1h,
                               hh_change_3h, overall_change_3h, hh_ratio, overall_ratio])
        return inputs

    def get_charge(self, remaining_steps, ar, **kwargs):

        inputs = self.basic_inputs(ar, remaining_steps)

        self.last_charging = ar.max_charging_speed*self.eval_network(inputs)[0]

        return self.last_charging

class AdvancedNNPlanner(NNPlanner):

    def __init__(self, *args, **kwargs):
        super(AdvancedNNPlanner, self).__init__(*args, **kwargs)

    def copy(self):
        nn = AdvancedNNPlanner(self.layer_sizes, self.activations, self.hh_only)
        nn.set_network(self.vectorized_net)
        return nn

    def inputs(self, remaining_steps, ar, current_time):
        t1, t2 = utils.encode_time(current_time)
        workday = 1.0 if current_time.weekday() < 5 else 0.0

        advanced_inputs = np.array([t1, t2, workday])
        basic_inputs = self.basic_inputs(ar, remaining_steps)

        inputs = np.append(advanced_inputs, basic_inputs)
        return inputs

    def get_charge(self, remaining_steps, ar, current_time = None):
        inputs = self.inputs(remaining_steps, ar, current_time)
        self.last_charging = ar.max_charging_speed * self.eval_network(inputs)[0]
        return self.last_charging

class AdvancedNNPlannerTime(AdvancedNNPlanner):

    def __init__(self, *args, **kwargs):
        super(AdvancedNNPlannerTime, self).__init__(*args, **kwargs)

    def copy(self):
        nn = AdvancedNNPlannerTime(self.layer_sizes, self.activations, self.hh_only)
        nn.set_network(self.vectorized_net)
        return nn

    def inputs(self, remaining_steps, ar, current_time):
        inputs = super().inputs(remaining_steps, ar, current_time)
        
        t1, t2 = utils.encode_time(ar.available_until)
        req_len = (ar.available_until - ar.available_after)/datetime.timedelta(hours=24)

        new_inputs = np.array([t1, t2, req_len])
        return np.append(new_inputs, inputs)

    def get_charge(self, remaining_steps, ar, current_time = None):
        inputs = self.inputs(remaining_steps, ar, current_time)
        self.last_charging = ar.max_charging_speed * self.eval_network(inputs)[0]
        return self.last_charging

class ESNPlanner:

    def __init__(self, n_reservoir, alpha, hh_only, recurrent=False):
        self.hh_only = hh_only
        self.n_inputs = 12 if hh_only else 17
        if recurrent:
            self.n_inputs += 1
        self.n_reservoir = n_reservoir
        self.alpha = alpha
        self.esn = esn.ESN(self.n_inputs, 1, n_reservoir, alpha)

        self.recurrent=recurrent

        self.last_hh_consumptions = []
        self.last_overall_consumptions = []
        self.last_charging_consumptions = []
        self.last_charging = 0

        self.last_states = []

    def copy(self):
        nn = ESNPlanner(self.n_reservoir, self.alpha, self.hh_only, self.recurrent)
        nn.esn.W_in = self.esn.W_in
        nn.esn.W = self.esn.W
        nn.esn.W_out = self.esn.W_out
        return nn

    def update_info(self, household_consumption, overall_consumption, *, current_time, ar = None, **kwargs):
        self.last_hh_consumptions.append(household_consumption + self.last_charging)
        self.last_overall_consumptions.append(overall_consumption)
        self.last_charging_consumptions.append(self.last_charging)

        # store only last 24 hours
        self.last_charging_consumptions = self.last_charging_consumptions[-2 * 24:]
        self.last_overall_consumptions = self.last_overall_consumptions[-2 * 24:]
        self.last_hh_consumptions = self.last_hh_consumptions[-2 * 24:]

        remaining_charge = ar.remaining_charge if ar else 0
        initial_charge = ar.initial_charge if ar else 1
        initial_steps = ar.initial_steps if ar else 1
        remaining_steps = utils.remaining_steps(ar, current_time) if ar else 1

        last_hh = np.array(self.last_hh_consumptions)
        last_oa = np.array(self.last_overall_consumptions)

        max_recent_hh = np.max(last_hh)
        avg_recent_hh = np.mean(last_hh)
        min_recent_hh = np.min(last_hh)
        max_recent_overall = np.max(last_oa)
        avg_recent_overall = np.mean(last_oa)
        min_recent_overall = np.min(last_oa)

        hh_consumption = 0
        overall_consumption = 0
        hh_change_last = 0
        overall_change_last = 0
        hh_change_1h = 0
        overall_change_1h = 0
        hh_change_3h = 0
        overall_change_3h = 0
        hh_ratio = 0
        overall_ratio = 0

        if avg_recent_hh > 0:
            hh_consumption = self.last_hh_consumptions[-1] / avg_recent_hh - 1
        if avg_recent_overall > 0:
            overall_consumption = self.last_overall_consumptions[-1] / avg_recent_overall - 1
        if max_recent_hh != min_recent_hh:
            hh_ratio = (self.last_hh_consumptions[-1] - min_recent_hh) / (max_recent_hh - min_recent_hh)
        if max_recent_overall != min_recent_overall:
            overall_ratio = (self.last_overall_consumptions[-1] - min_recent_overall) / (
                    max_recent_overall - min_recent_overall)
        if len(self.last_hh_consumptions) > 1 and self.last_hh_consumptions[-2] > 0:
            hh_change_last = self.last_hh_consumptions[-1] / self.last_hh_consumptions[-2] - 1
            overall_change_last = self.last_overall_consumptions[-1] / self.last_overall_consumptions[-2] - 1
        if len(self.last_hh_consumptions) > 2 and self.last_hh_consumptions[-3] > 0:
            hh_change_1h = self.last_hh_consumptions[-1] / self.last_hh_consumptions[-3] - 1
            overall_change_1h = self.last_overall_consumptions[-1] / self.last_overall_consumptions[-3] - 1
        if len(self.last_hh_consumptions) > 6 and self.last_hh_consumptions[-7] > 0:
            hh_change_3h = self.last_hh_consumptions[-1] / self.last_hh_consumptions[-7] - 1
            overall_change_3h = self.last_overall_consumptions[-1] / self.last_overall_consumptions[-7] - 1

        const_ch_speed = 0
        min_ch_speed = 0

        if ar:
            const_ch_speed = utils.constant_charging_speed(remaining_steps, ar) / ar.max_charging_speed
            min_ch_speed = utils.minimum_charging_speed(remaining_steps, ar) / ar.max_charging_speed

        t1, t2 = utils.encode_time(current_time)
        workday = 1.0 if current_time.weekday() < 5 else 0.0

        inputs_list = []
        if self.hh_only:
            inputs_list = [remaining_charge / initial_charge, remaining_steps / initial_steps,
                           const_ch_speed, min_ch_speed, hh_consumption, hh_change_last, hh_change_1h,
                           hh_change_3h, hh_ratio, t1, t2, workday]
        else:
            inputs_list = [remaining_charge / initial_charge, remaining_steps / initial_steps,
                           const_ch_speed, min_ch_speed, hh_consumption, overall_consumption,
                           hh_change_last, overall_change_last, hh_change_1h, overall_change_1h,
                           hh_change_3h, overall_change_3h, hh_ratio, overall_ratio, t1, t2, workday]

        if (self.recurrent):
            inputs_list.append(self.last_charging/ar.max_charging_speed if ar else 0)

        inputs = np.array(inputs_list)

        nn_output = utils.sigmoid(self.esn.update(inputs))[0]

        max_charging_speed = ar.max_charging_speed if ar else 0
        self.last_charging = max_charging_speed * max(min_ch_speed, nn_output) if ar else 0

        # self.last_states.append((np.concatenate([np.insert(inputs, 0, 1.0), self.esn.state]), nn_output,
        #                                       min_ch_speed, max_charging_speed))   # save information for gradient

    def get_charge(self, remaining_steps, ar, **kwargs):
        return self.last_charging

    def set_network(self, vectorized_net):
        self.esn.W_out = np.reshape(vectorized_net, newshape=self.esn.W_out.shape)

    def set_additional_params(self, additional_params):
        self.esn.W_in = additional_params['W_in']
        self.esn.W = additional_params['W']

    def vectorized_size(self):
        return self.esn.W_out.size

class EnsemblePlanner:

    def __init__(self, planners):
        self.planners = planners

    def update_info(self, household_consumption, overall_consumption, *, current_time, ar=None, **kwargs):
        for p in self.planners:
            p.update_info(household_consumption, overall_consumption, current_time=current_time, ar=ar, **kwargs)

    def get_charge(self, remaining_steps, ar, **kwargs):
        charges = [p.get_charge(remaining_steps, ar, **kwargs) for p in self.planners]
        return np.mean(charges)

    def copy(self):
        return EnsemblePlanner([p.copy(houseid) for p in self.planners])

import pandas as pd
class OptimumPlanner(AdvancedNNPlannerTime):

    def __init__(self, optimum_charges, *args, **kwargs):
        super(OptimumPlanner, self).__init__([12, 5, 1], [utils.relu, utils.sigmoid], False)
        self.set_network(np.random.uniform(size=self.vectorized_size()))
        self.optimum_charges = optimum_charges
        self.training_data = []

    def copy(self):
        return OptimumPlanner(self.optimum_charges)

    #def update_info(self, household_consumption, overall_consumption, **kwargs):
    #    pass

    def write_training_data(self, houseid):
        df = pd.DataFrame(self.training_data)
        df.columns = ['end_t1', 'end_t2', 'req_len', 'cur_t1', 'cur_t2', 
                      'workday','rem_ch', 'rem_steps', 'const_ch', 
                      'min_ch', 'hh_cons', 'ov_cons', 'hh_change', 'ov_change', 
                      'hh_change_1h', 'ov_change_1h', 'hh_change_3h', 
                      'ov_change_3h', 'hh_ratio', 'ov_ratio', 'target']
        df.to_csv(f'../train/{houseid}.csv', index=False)

    def get_charge(self, remaining_steps, ar, current_time = None):
        inputs = self.inputs(remaining_steps, ar, current_time)
        output = self.optimum_charges[ar.houseid][current_time]
        self.training_data.append(np.append(inputs, output/ar.max_charging_speed))
        return output

class SavedPlanner(AdvancedNNPlannerTime):

    def __init__(self, file_name, layer_sizes, activations, hh_only=False):
        super(SavedPlanner, self).__init__(layer_sizes, activations, hh_only) # the planner is never used - we only need the inputs
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.file_name = file_name
        self.hh_only = hh_only
        self.set_network(np.load(file_name))

    def copy(self):
        return SavedPlanner(self.file_name, self.layer_sizes, self.activations, self.hh_only)

    def get_charge(self, remaining_steps, ar, current_time=None):
        charge = super().get_charge(remaining_steps, ar, current_time=current_time)
        return charge