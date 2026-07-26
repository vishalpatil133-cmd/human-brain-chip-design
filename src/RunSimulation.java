package src;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.ArrayList;

/**
 * Main simulation driver. Runs a 25,000-neuron SNN simulation with 1% connection sparsity.
 * Outputs compact JSON results representing spike events and metrics.
 */
public class RunSimulation {
    private static final int STEPS = 100;
    private static final int NUM_INPUTS = 25000;
    private static final int NUM_NEURONS = 25000;
    private static final double SYNAPSE_DENSITY = 0.01; // 1% connectivity

    public static void main(String[] args) {
        System.out.println("Initializing 25K Neuron Sparse SNN Simulation...");

        // Create sparse crossbar: threshold = 255, leak = 3, reset = 0
        CrossbarSimulator simulator = new CrossbarSimulator(
            NUM_INPUTS, NUM_NEURONS, 255, 3, 0, SYNAPSE_DENSITY, 42
        );

        Random rand = new Random(100);
        
        // Generate sparse input spike trains (0.4% average spike rate per input per step)
        boolean[][] inputSpikeTrains = new boolean[STEPS][NUM_INPUTS];
        for (int t = 0; t < STEPS; t++) {
            // About 100 random inputs spike in each cycle
            for (int k = 0; k < 100; k++) {
                int ch = rand.nextInt(NUM_INPUTS);
                inputSpikeTrains[t][ch] = true;
            }
        }

        // Build compact JSON output
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"numSteps\": ").append(STEPS).append(",\n");
        json.append("  \"numNeurons\": ").append(NUM_NEURONS).append(",\n");
        json.append("  \"steps\": [\n");

        for (int t = 0; t < STEPS; t++) {
            boolean[] inputs = inputSpikeTrains[t];
            boolean[] outputs = simulator.step(inputs);

            json.append("    {\n");
            json.append("      \"step\": ").append(t).append(",\n");
            
            // Record ONLY indices of active inputs (extremely compact)
            json.append("      \"activeInputs\": [");
            ArrayList<Integer> actIns = new ArrayList<>();
            for (int i = 0; i < NUM_INPUTS; i++) {
                if (inputs[i]) actIns.add(i);
            }
            for (int i = 0; i < actIns.size(); i++) {
                json.append(actIns.get(i));
                if (i < actIns.size() - 1) json.append(",");
            }
            json.append("],\n");

            // Record ONLY indices of active outputs
            json.append("      \"activeOutputs\": [");
            ArrayList<Integer> actOuts = new ArrayList<>();
            for (int i = 0; i < NUM_NEURONS; i++) {
                if (outputs[i]) actOuts.add(i);
            }
            for (int i = 0; i < actOuts.size(); i++) {
                json.append(actOuts.get(i));
                if (i < actOuts.size() - 1) json.append(",");
            }
            json.append("],\n");

            // Detailed potentials of the first 16 neurons for plotting
            json.append("      \"potentials16\": [");
            NeuronModel[] neurons = simulator.getNeurons();
            for (int i = 0; i < 16; i++) {
                json.append(neurons[i].getPotential());
                if (i < 15) json.append(",");
            }
            json.append("]\n");

            json.append("    }");
            if (t < STEPS - 1) json.append(",");
            json.append("\n");
        }

        json.append("  ],\n");

        // Energy summaries
        double energyDense = simulator.getDenseEnergy();
        double energyNeuromorphic = simulator.getNeuromorphicEnergy(STEPS);
        double energySavings = (1.0 - (energyNeuromorphic / energyDense)) * 100.0;

        json.append("  \"energyDense\": ").append(energyDense).append(",\n");
        json.append("  \"energyNeuromorphic\": ").append(energyNeuromorphic).append(",\n");
        json.append("  \"energySavingsPercent\": ").append(String.format("%.2f", energySavings)).append(",\n");
        json.append("  \"denseOps\": ").append(simulator.getTotalDenseOps()).append(",\n");
        json.append("  \"synapticOps\": ").append(simulator.getTotalSynapticOps()).append("\n");
        json.append("}\n");

        // Write JSON results
        try (FileWriter file = new FileWriter("simulation_results.json")) {
            file.write(json.toString());
        } catch (IOException e) {
            System.err.println("Error writing simulation_results.json: " + e.getMessage());
        }

        // Write JS results to bypass CORS
        try (FileWriter file = new FileWriter("simulation_results.js")) {
            file.write("const SIMULATION_DATA = " + json.toString() + ";");
            System.out.println("25K Sparse SNN Simulation completed successfully.");
            System.out.println("Saved results to 'simulation_results.json' and 'simulation_results.js'.");
            System.out.printf("--- Energy Report ---\n");
            System.out.printf("Conventional Accelerator Energy: %.1f units\n", energyDense);
            System.out.printf("Neuromorphic Core Energy:        %.1f units\n", energyNeuromorphic);
            System.out.printf("Energy Savings:                  %.2f%%\n", energySavings);
            System.out.printf("Total Sparse Synaptic ops:       %d ops\n", simulator.getTotalSynapticOps());
            System.out.printf("Total Dense operations avoided:   %d ops\n", simulator.getTotalDenseOps());
        } catch (IOException e) {
            System.err.println("Error writing simulation_results.js: " + e.getMessage());
        }
    }
}
