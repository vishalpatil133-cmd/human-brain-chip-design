package src;

import java.util.Random;

/**
 * Encodes 4x4 pixel images into Poisson-distributed spike trains for SNN input.
 */
public class ImagePreprocessor {
    private final Random rand;

    public ImagePreprocessor(long seed) {
        this.rand = new Random(seed);
    }

    /**
     * Converts a 4x4 image (intensities 0.0 to 1.0) into a spike train over a given duration.
     * @param image 4x4 grid of pixel values [0.0 - 1.0].
     * @param steps Total time steps of the spike train.
     * @return 2D boolean array [steps][16] representing input spikes over time.
     */
    public boolean[][] encode(double[][] image, int steps) {
        boolean[][] spikeTrain = new boolean[steps][16];
        
        for (int t = 0; t < steps; t++) {
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    int channel = r * 4 + c;
                    double intensity = image[r][c];
                    
                    // Poisson Rate Coding: probability of spiking is proportional to intensity
                    if (rand.nextDouble() < intensity) {
                        spikeTrain[t][channel] = true;
                    }
                }
            }
        }
        
        return spikeTrain;
    }
}
