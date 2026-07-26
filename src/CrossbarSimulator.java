package src;

import java.util.Random;

/**
 * Simulates a large-scale synaptic crossbar using an optimized sparse adjacency list
 * representation (1% connection density) to support 25,000 neurons efficiently.
 */
public class CrossbarSimulator {
    private final int numInputs;
    private final int numNeurons;
    
    // Adjacency list representation of sparse synaptic weights:
    // synapseDestinations[pre] = array of target neuron indices
    // synapseWeights[pre]      = array of corresponding weight values
    private final int[][] synapseDestinations;
    private final int[][] synapseWeights;
    
    private final NeuronModel[] neurons;

    // Energy modeling constants (arbitrary units proportional to picojoules - pJ)
    public static final double ENERGY_DENSE_MAC = 1.5;     // Energy per MAC in traditional dense hardware
    public static final double ENERGY_NEUROMORPHIC_SYNAPSE = 0.1; // Energy per sparse synaptic addition
    public static final double ENERGY_NEUROMORPHIC_LEAK = 0.05;   // Base leakage update energy per neuron

    private long totalSynapticOps = 0; // Neuromorphic active operations
    private long totalDenseOps = 0;    // Conventional dense matrix operations

    public CrossbarSimulator(int numInputs, int numNeurons, int threshold, int leak, int resetPotential, double density, long seed) {
        this.numInputs = numInputs;
        this.numNeurons = numNeurons;
        this.neurons = new NeuronModel[numNeurons];

        for (int i = 0; i < numNeurons; i++) {
            this.neurons[i] = new NeuronModel(threshold, leak, resetPotential);
        }

        // Initialize sparse connectivity
        Random rand = new Random(seed);
        this.synapseDestinations = new int[numInputs][];
        this.synapseWeights = new int[numInputs][];

        int connectionsPerInput = (int) (numNeurons * density);
        if (connectionsPerInput < 1) connectionsPerInput = 1;

        for (int pre = 0; pre < numInputs; pre++) {
            synapseDestinations[pre] = new int[connectionsPerInput];
            synapseWeights[pre] = new int[connectionsPerInput];
            
            // Randomly connect this input to 'connectionsPerInput' neurons
            for (int k = 0; k < connectionsPerInput; k++) {
                synapseDestinations[pre][k] = rand.nextInt(numNeurons);
                
                // Excitatory weights (positive values 20 to 120)
                synapseWeights[pre][k] = 20 + rand.nextInt(100); 
            }
        }
    }

    /**
     * Advances the neuromorphic core by one time-step.
     * @param inputSpikes Boolean array indicating which input channels spiked.
     * @return Boolean array of output spikes.
     */
    public boolean[] step(boolean[] inputSpikes) {
        boolean[] outputSpikes = new boolean[numNeurons];

        // 1. Process active input spikes (synaptic integration) using adjacency list
        for (int pre = 0; pre < numInputs; pre++) {
            if (inputSpikes[pre]) {
                int[] targets = synapseDestinations[pre];
                int[] weights = synapseWeights[pre];
                int len = targets.length;
                
                for (int k = 0; k < len; k++) {
                    neurons[targets[k]].integrate(weights[k]);
                    totalSynapticOps++; // Sparse active operation
                }
            }
        }

        // Traditional dense architectures calculate ALL possible connections
        totalDenseOps += (long) numInputs * numNeurons;

        // 2. Apply leakage, check spikes, and update neuron potentials
        for (int post = 0; post < numNeurons; post++) {
            neurons[post].decay();
            outputSpikes[post] = neurons[post].checkSpike();
        }

        return outputSpikes;
    }

    public NeuronModel[] getNeurons() {
        return neurons;
    }

    public long getTotalSynapticOps() {
        return totalSynapticOps;
    }

    public long getTotalDenseOps() {
        return totalDenseOps;
    }

    public double getDenseEnergy() {
        return totalDenseOps * ENERGY_DENSE_MAC;
    }

    public double getNeuromorphicEnergy(int totalTimeSteps) {
        double synapseEnergy = totalSynapticOps * ENERGY_NEUROMORPHIC_SYNAPSE;
        double leakEnergy = (double) totalTimeSteps * numNeurons * ENERGY_NEUROMORPHIC_LEAK;
        return synapseEnergy + leakEnergy;
    }

    public void reset() {
        totalSynapticOps = 0;
        totalDenseOps = 0;
        for (NeuronModel neuron : neurons) {
            neuron.reset();
        }
    }
}
