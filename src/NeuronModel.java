package src;

/**
 * Models a single Leaky Integrate-and-Fire (LIF) Neuron matching the RTL implementation.
 */
public class NeuronModel {
    private int potential;
    private final int threshold;
    private final int leak;
    private final int resetPotential;
    private int spikeCount;

    public NeuronModel(int threshold, int leak, int resetPotential) {
        this.potential = resetPotential;
        this.threshold = threshold;
        this.leak = leak;
        this.resetPotential = resetPotential;
        this.spikeCount = 0;
    }

    /**
     * Integrates weighted input spike energy.
     */
    public void integrate(int weight) {
        this.potential += weight;
    }

    /**
     * Simulates the leak of membrane potential towards the reset potential.
     */
    public void decay() {
        if (this.potential > this.resetPotential) {
            this.potential = Math.max(this.resetPotential, this.potential - this.leak);
        } else if (this.potential < this.resetPotential) {
            this.potential = Math.min(this.resetPotential, this.potential + this.leak);
        }
    }

    /**
     * Checks if the neuron crosses the threshold to spike.
     * @return true if a spike is fired, false otherwise.
     */
    public boolean checkSpike() {
        if (this.potential >= this.threshold) {
            this.potential = this.resetPotential;
            this.spikeCount++;
            return true;
        }
        return false;
    }

    public int getPotential() {
        return this.potential;
    }

    public int getSpikeCount() {
        return this.spikeCount;
    }

    public void reset() {
        this.potential = this.resetPotential;
        this.spikeCount = 0;
    }
}
