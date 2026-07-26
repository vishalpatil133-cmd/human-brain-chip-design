package src;

import java.util.Random;

/**
 * Encodes audio frequency sweeps into temporal spike sequences for SNN input.
 * Simulates a simple silicon cochlea filter bank.
 */
public class AudioPreprocessor {
    private final Random rand;

    public AudioPreprocessor(long seed) {
        this.rand = new Random(seed);
    }

    /**
     * Generates a frequency sweep spike train.
     * @param rising If true, generates a rising pitch sweep (low channels to high). If false, falling sweep.
     * @param steps Total time steps of the audio sample.
     * @return 2D boolean array [steps][16] representing frequency spikes over time.
     */
    public boolean[][] encodeSweep(boolean rising, int steps) {
        boolean[][] spikeTrain = new boolean[steps][16];
        int quarter = steps / 4;
        
        for (int t = 0; t < steps; t++) {
            // Determine active frequency band based on current time
            int activeBand = t / quarter; // 0, 1, 2, 3
            if (!rising) {
                activeBand = 3 - activeBand; // Reverse order for falling pitch
            }
            
            // Map active band to 4 input channels
            int channelStart = activeBand * 4;
            
            for (int ch = 0; ch < 16; ch++) {
                if (ch >= channelStart && ch < channelStart + 4) {
                    // Active frequency band has high chance of spiking
                    if (rand.nextDouble() < 0.8) {
                        spikeTrain[t][ch] = true;
                    }
                } else {
                    // Background noise spikes occasionally
                    if (rand.nextDouble() < 0.05) {
                        spikeTrain[t][ch] = true;
                    }
                }
            }
        }
        
        return spikeTrain;
    }
}
