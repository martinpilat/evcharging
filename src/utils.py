import numpy
import math

import settings


def minimum_charging_speed(remaining_steps, active_request):
    charge_step = active_request.max_charging_speed / 2
    return 2*max(0.0, active_request.remaining_charge - (remaining_steps - 1)*charge_step)

def constant_charging_speed(remaining_steps, active_request):
    charge_step = active_request.remaining_charge/remaining_steps
    return 2*charge_step

def relu(x):
    return numpy.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + numpy.exp(-numpy.clip(x, -15, 15)))

def ident(x):
    return x

def remaining_steps(ar, current_time):
    return (ar.available_until - current_time)//settings.TIME_STEP

def encode_time(current_time):
    seconds = current_time.hour*3600+current_time.minute*60
    day_fraction = seconds/(24*3600)
    return math.cos(2*math.pi*day_fraction), math.sin(2*math.pi*day_fraction)