import numpy as np
import logging
import itertools
import warnings
from scipy import sparse


def generateSinusInput(length):
    """
    @desc: Generates a sinus signal (one 'hill') in given length of time steps
    """
    # Draw from a sin wave from 0 to 3,14 (one 'hill')
    probCoeff = 1  # 1.5  # the higher the probability coefficient, the more activity is in the network
    probs = probCoeff * np.abs(np.sin((np.pi / length) * np.arange(length)))
    randoms = np.random.rand(length)

    # Get indices of spike
    spikeInd = np.where(randoms < probs)[0]

    # Return spikes
    return spikeInd


def generateUniformInput(length, prob=0.1):
    """
    @desc: Generates a simple input signal
    """
    spikes = np.zeros((length, 1))
    randoms = np.random.rand(length)

    # Draw spikes
    for i in range(length):
        spikes[i] = (randoms[i] < prob)

    # Get indices of spike
    spikeTimes = np.where(spikes)[0]

    # Return spikes
    return spikeTimes


def addInput(self, *args, **kwargs):
    """
    @desc:  Adds an input to the network for every trial,
            either a sequence or a single input
    """
    print(self.p.inputIsSequence)
    if self.p.inputIsSequence:  # False
        addInputSequence(self, *args, **kwargs)
    else:
        addInputSingle(self, *args, **kwargs)


def addInputSequence(self):
    """
    @desc:  Create a sequence of inputs
    NOTE a topology is currently not supported in a sequence input
    """
    if self.p.inputIsTopology:
        warnings.warn("inputIsTopology is currently not supported for an input sequence and will be ignored")

    # Get number of generators
    num_gens = self.p.inputNumTargetNeurons

    # Iterate over number of inputs
    for i in range(self.p.inputSequenceSize):
        # Create spike generator
        sg = self.nxNet.createSpikeGenProcess(numPorts=num_gens)

        # Draw spikes for input generators for current input
        input_spikes = drawSpikesForAllGenerators(self, num_gens, offset=i * self.p.inputSteps)
        self.inputSpikes.append(input_spikes)

        # Add spikes s to generator i
        for k, s in enumerate(input_spikes):
            if type(s) is not list: s = s.tolist()
            sg.addSpikes(spikeInputPortNodeIds=k, spikeTimes=s)

        # Get inidices of target neurons for current input
        input_target_neurons = np.arange(i * self.p.inputNumTargetNeurons, (i + 1) * self.p.inputNumTargetNeurons)
        self.inputTargetNeurons.append(input_target_neurons)

        # Connect spike generators to reservoir
        self.inputWeights = connectSpikeGenerator(self, sg, input_target_neurons)

    # Log that input was added
    logging.info('Input sequence was added to the network')


def addInputSingle(self, input_spike_indices=None, target_neuron_indices=None):
    """
    @desc:  Connects a single input per trial to the reservoir network
    @note:  If inputIsTopology is true the number of target neurons may differ
            due to rounding in getTargetNeurons() function
            therefore self.p.inputNumTargetNeurons cannot be used here,
            but instead len(self.inputTargetNeurons) must be used
    @params:
            inputSpikeIndices:      indices of input spikes for spike generators
                                    if not given, they are drawn (default)
            targetNeuronIndices:    indices of reservoir neurons to connect input to
                                    if not given, indices are taken successively (default)
    """
    # Get indices of target neurons if not already given
    if target_neuron_indices is None:
        target_neuron_indices = []
    if input_spike_indices is None:
        input_spike_indices = []
    self.inputTargetNeurons = target_neuron_indices if len(target_neuron_indices) else getTargetNeurons(self)
    print('self.inputTargetNeurons: {}'.format(self.inputTargetNeurons))

    # Get number of generators
    num_gens = len(self.inputTargetNeurons)
    # print('num_gens: {}'.format(num_gens))

    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=num_gens)
    # print('sg: {}'.format(sg))

    # Draw spikes for input generators if not already given
    self.inputSpikes = input_spike_indices if len(input_spike_indices) else drawSpikesForAllGenerators(self,
                                                                                                       numGens=num_gens)
    # print('self.inputSpikes: {}'.format(self.inputSpikes))

    # Add spikes s to generator i
    for i, s in enumerate(self.inputSpikes):
        if type(s) is not list:
            s = s.tolist()
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=s)

    # Connect spike generators to reservoir
    self.inputWeights = connectSpikeGenerator(self, sg, self.inputTargetNeurons)

    # Log that input was added
    logging.info('Input was added to the network')


def drawSpikesForAllGenerators(self, numGens, offset=0):
    """
    @desc: Draw spikes for ALL spike generators
    """
    # Initialize array for spike indices
    input_spikes = []

    # Define empty variable for combinations
    combinations = None
    # If leave out should be applied, define all possible combinations to leave one out
    if self.p.inputIsLeaveOut:
        combinations = np.array(
            list(itertools.combinations(np.arange(len(self.inputTargetNeurons)), self.p.inputNumLeaveOut)))

    # Iterate over target neurons to draw spikes
    for i in range(numGens):
        # Initialize array for spike indices for generator i
        spike_times = []

        # Defines spikes for generator i for all trials
        for k in range(self.p.trials):
            apply = True

            # If leave out should be applied, update apply boolean
            if self.p.inputIsLeaveOut:
                # Add spike times only when i != k
                apply = np.all([combinations[k, m] != i for m in range(self.p.inputNumLeaveOut)])

            # If spike generator produces input for the current trial, add it to spike_times
            if apply:
                off = offset + self.p.stepsPerTrial * k + self.p.resetOffset * (k + 1)
                spks = drawSpikes(self, offset=off)
                spike_times.append(spks)

        # Add spike indices to input_spikes array
        input_spikes.append(list(itertools.chain(*spike_times)))

    return input_spikes


def drawSpikes(self, offset=0):
    """
    @desc: Draw spikes for ONE spike generator
    """
    s = []
    # Generate spikes, depending on input type
    if self.p.inputType == 'uniform':
        s = self.p.inputOffset + offset + generateUniformInput(self.p.inputSteps, prob=self.p.inputGenSpikeProb)
    if self.p.inputType == 'sinus':
        s = self.p.inputOffset + offset + generateSinusInput(self.p.inputSteps)

    return s


def getTargetNeurons(self):
    """
    @desc: Define target neurons in reservoir to connect generators with
    """
    # Initialize array for target neurons
    target_neurons = []

    # If topology should NOT be considered, just take first n neurons as target
    if not self.p.inputIsTopology:
        target_neurons = np.arange(self.p.inputNumTargetNeurons)

    # If topology should be considered, define square input target area
    if self.p.inputIsTopology:
        # Define size and 
        target_neurons_edge = int(np.sqrt(self.p.inputNumTargetNeurons))
        ex_neurons_edge = int(np.sqrt(self.p.reservoirExSize))

        # Get shifts for the input area of the target neurons
        sX = self.p.inputShiftX
        sY = self.p.inputShiftY

        # Define input region in network topology and store their indices
        topology = np.zeros((ex_neurons_edge, ex_neurons_edge))
        topology[sY:sY + target_neurons_edge, sX:sX + target_neurons_edge] = 1
        target_neurons = np.where(topology.flatten())[0]

    return target_neurons


def connectSpikeGenerator(self, spikeGenerators, inputTargetNeurons):
    """
    @desc:  Connect spike generators with target neurons of reservoir
            Every spike generator is connected to one neuron
            Finally draws uniformly distributed weights
    @params:
            spikeGenerators: nxsdk spike generators
            inputTargetNeurons: indices of reservoir target neurons
    """
    # Creates empty mask matrix
    input_mask = np.zeros((self.p.reservoirExSize, len(inputTargetNeurons)))

    # Every generator is connected to one 
    for i, idx in enumerate(inputTargetNeurons):
        input_mask[idx, i:i + 1] = 1

    # Transform to sparse matrix
    input_mask = sparse.csr_matrix(input_mask)

    # Draw weights
    input_weights = self.drawSparseWeightMatrix(input_mask, distribution='uniform')

    # Connect generator to the reservoir network
    for i in range(len(self.exReservoirChunks)):
        fr, to = i * self.p.neuronsPerCore, (i + 1) * self.p.neuronsPerCore
        ma = input_mask[fr:to, :].toarray()
        we = input_weights[fr:to, :].toarray()
        spikeGenerators.connect(self.exReservoirChunks[i], prototype=self.genConnProto, connectionMask=ma, weight=we)

    return input_weights
