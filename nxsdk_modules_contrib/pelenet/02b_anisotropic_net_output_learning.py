"""
A network with an inhomogeneous weight structure (anisotropic network)

This example contains a reservoir network of 4500 neurons, distributed over 2 Loihi chips.
3600 neurons are excitatory and 900 neurons are inhibitory.
The neurons are locally connected with a gaussian distribution on a 2-dimensional grid of neurons.
Edges are connected to form a torus shape.
The gaussian distribution is shifted, where the shift direction is drawn by perlin noise.

The network activity is started with a short cue input and then maintained.
After some time the network activity is stopped using a snip.
The next trial is started with a short cue input again,
which is slightly different (noisy) than the one before.

Using linear regression, a function is trained from the trials (using all but the last).
Finally the spiking activity of the last trial is used to predict the function,
using the weights trained with the other trials.

The experiment is defined in 'pelenet/experiments/readoutanisotropic.py' file.
A log file, parameters, and plot figures are stored in the 'log' folder for every run of the simulation.

NOTE: The main README file contains some more information about the structure of pelenet
"""

# Load pelenet modules
from pelenet.utils import Utils
from pelenet.experiments.readoutanisotropic import AnisotropicReadoutExperiment

# Official modules
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['SLURM'] = '1'
os.environ['PARTITION'] = 'nahuku32'


def main():
    # Overwrite default parameters (pelenet/parameters/ and pelenet/experiments/readoutanisotropic.py)
    parameters = {
        # Experiment
        'seed': 3,  # Random seed
        'trials': 25,  # Number of trials
        'stepsPerTrial': 110,  # Number of simulation steps for every trial
        'isReset': True,  # Activate reset after every trial
        # Network
        'refractoryDelay': 2,  # Refactory period
        'voltageTau': 10.24,  # Voltage time constant
        'currentTau': 10.78,  # Current time constant
        'thresholdMant': 1000,  # Spiking threshold for membrane potential
        'reservoirConnProb': 0.05,
        # Anisotropic
        'anisoStdE': 12,  # Space constant, std of gaussian for excitatory neurons
        'anisoStdI': 9,  # Space constant, std of gaussian for inhibitory neurons (range 9 - 11)
        'anisoShift': 1,  # Intensity of the shift of the connectivity distribution for a neuron
        # 'percShift': 1,  # Percentage of shift (default 1)
        'anisoPerlinScale': 4,  # Perlin noise scale, high value => dense valleys, low value => broad valleys
        'weightExCoefficient': 12,  # Coefficient for excitatory anisotropic weight
        'weightInCoefficient': 48,  # Coefficient for inhibitory anisotropic weight
        # Input
        'inputIsTopology': True,  # Activate a 2D input area
        'inputIsLeaveOut': True,  # Leaves one target neuron out per trial
        'patchNeuronsShiftX': 44,  # x-position of the input area
        'patchNeuronsShiftY': 24,  # y-position of the input area
        'inputNumTargetNeurons': 25,  # Number of target neurons for the input
        'inputSteps': 5,  # Number of steps the network is activated by the input
        'inputWeightExponent': 0,  # The weight exponent of the weights from the generator to the target neurons
        'inputGenSpikeProb': 1.0,  # Spiking probability of the spike generators
        # Output
        'partitioningClusterSize': 10,  # Size of clusters connected to an output neuron (6|10)
        # Probes
        'isExSpikeProbe': True,  # Probe excitatory spikes
        'isInSpikeProbe': True,  # Probe inhibitory spikes
        'isOutSpikeProbe': True  # Probe output spikes
    }

    # Initilizes the experiment, also initializes the log
    # Creating a new object results in a new log entry in the 'log' folder
    # The name is optional, it is extended to the folder in the log directory
    exp = AnisotropicReadoutExperiment(name='anisotropic-network-output-learning', parameters=parameters)

    # Instantiate the utils singleton
    utils = Utils.instance()

    # Build the network, in this function the weight matrix, inputs, probes, etc. are defined and created
    exp.build()
    # return

    # Run the network simulation, afterwards the probes are postprocessed to nice arrays
    exp.run()

    # Get spike data from excitatory neurons and pool neurons
    dataEx = exp.net.condenseSpikeProbes(exp.net.exSpikeTrains)
    dataOut = exp.net.condenseSpikeProbes(exp.net.outSpikeTrains)

    # Define target
    target = np.sin(np.arange(0, 2, 0.02))
    print(dataEx.shape)
    print(dataOut.shape)
    print(target.shape)

    # Prepare excitatory spiking data
    (x, xe, y) = utils.prepareDataset(dataEx, target, binSize=10)
    print(x.shape)
    print(xe.shape)
    print(y.shape)

    # Estimate parameters based all but last trial and predict target based on last trial
    # Apply elastic net estimation approach to reduce parameters
    predEx = utils.estimateMovement(x, xe, y, alpha=0.001, L1_wt=0.05)
    print(predEx.shape)

    # Prepare pool spiking data
    (x, xe, y) = utils.prepareDataset(dataOut, target, binSize=10)

    # Estimate parameters based all but last trial and predict target based on last trial
    predPool = utils.estimateMovement(x, xe, y)
    print(predPool.shape)


if __name__ == '__main__':
    main()
