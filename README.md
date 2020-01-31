# Evolution and imitation learning of controllers for EV charging

This project contains the implementation of the evolution of controllers for the charging of electric vehicles. It is divided into several files which are briefly described here.

## Requirements

The project requires Python 3 (tested on 3.6) with `numpy`, `pandas`, `deap` (only for evolution), and `matplotlib` (only for generating of the plots), `cvxpy` (for quadratic optimization), `tensorflow` (for the imitation learning part).

## How to use it

- __simulation__ - start the simulator by running `python simulator.py`, the type of planner and the time period can be set in the `__main__` block; the log from the simulation can be accessed in the `sim.simulation_log` attribute and is a list of triples `(time, overall charging consumption, overall baseline consumption)`
- __evolutionary training__ - start the evolution by running `python evolution.py -c CONFIG`, where `CONFIG` is the name of the configuration file the fitness is defined as `evaluate_fitness`
- __gradient training__ - start the gradient training by running `python gradient.py -c CONFIG`, where `CONFIG` is the name of the configuration file

## The source codes

- `planners.py` -- contains the implementation of the planners (controllers)
- `simulator.py` -- the simulator of the charging of vehicles, simulates the grid in 30 minute steps (configurable in `settings.py`)
- `evolution.py` -- the implementation of the evolution of controllers based on neural networks, evolves the weights  of the neural network
- `gradient.py` -- implementation of the gradient based training of the models
- `utils.py` -- simple utilities - computes the minimum and constant charging speed, also contains the implementation of the activation functions for the neural networks
- `settings.py` -- global settings for the simulation (currently only `TIME_STEP`, which is however also hardcoded in some places, DO NOT CHANGE)
- `esn.py` -- simple implementation of the Echo State Network, used by some of the controllers

## The data

The input data are stored in the `data` folder. These contain a few files. In general, you should only need to use the first two in the list, the others were used to generate the charging requests from the two data sources.

- `requests_1.csv` -- generated charging requests based on the electricity consumption data from [UK Power Networks][1] and [National Household Travel Survey][2]
- `selected_1.csv` -- the baseline consumptions of the households that are simulated
- `baseline_1.csv` -- the sum of baseline consumptions of all other (not-simulated) households (450 in total)
- `trips.csv` -- the trips generated from the NHTS files
- `consumptions.csv` -- the consumptions of all 500 households

## Jupyter Notebooks

Some of the work (mainly data preparation) was done in Jupyter notebooks. 

The preparation of the consumption data is available in the `data/ProcessData.ipynb` notebook, and the charging requests are created in the `data/CreateRequests.ipynb` notebook.

The optimum charging schedules are computed in the `data/PrepareTraining.ipynb` notebook. The training sets can then be created by running the file `src/simulator.py` and simulating the optimal planner with the optimal charges saved in the file created by the notebook. The optimal planner writes all the training files into a single directory. 

Finally, the imitation learning is implemented in the `TrainModel.ipynb` notebook that was used in Google Colab to train the models and save the weights as `numpy` arrays.

The models can be again tested using the `src/simulator.py` file by using the `SavedModelPlanner`.

## Questions/bugs 

If you have any question, or if you find a bug, or have an idea for improvement, you can use the issues tracker or contact the author on Name.Surname at mff.cuni.cz.

[1]: https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households
[2]: https://nhts.ornl.gov/download.shtml#2009