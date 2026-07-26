// =============================================================================
// File: main_os.c
// Description: Minimal bare-metal Embedded C Firmware (OS/Bootloader) for the
//              custom RISC-V RV32I processor in the Neuromorphic SoC.
//              Handles hardware configurations, synapse programming, and
//              event-driven polling loops.
// Architect: Embedded Systems Firmware Engineer
// =============================================================================

#include <stdint.h>

// MMIO Base Addresses
#define SYNAPSE_RAM_BASE       0x00010000
#define NEURO_MMIO_BASE        0x00020000
#define ACCEL_MMIO_BASE        0x00040000

// MMIO Register Offsets
#define NEURO_REG_CTRL          ((volatile uint32_t*)(NEURO_MMIO_BASE + 0x00))
#define NEURO_REG_THRESHOLD     ((volatile uint32_t*)(NEURO_MMIO_BASE + 0x04))
#define NEURO_REG_LEAK          ((volatile uint32_t*)(NEURO_MMIO_BASE + 0x08))
#define NEURO_REG_INPUT_SPIKES  ((volatile uint32_t*)(NEURO_MMIO_BASE + 0x10)) // 8 words (256-bit)
#define NEURO_REG_OUT_SPIKES    ((volatile uint32_t*)(NEURO_MMIO_BASE + 0x30)) // 8 words (256-bit)
#define NEURO_REG_STDP_CTRL     ((volatile uint32_t*)(NEURO_MMIO_BASE + 0x50)) // V2 STDP control register

// CPU Local Controller Offsets (LPDDR5 / MMU)
#define ACCEL_REG_LPDDR5_CTRL   ((volatile uint32_t*)(ACCEL_MMIO_BASE + 0x10))
#define ACCEL_REG_LPDDR5_CFG    ((volatile uint32_t*)(ACCEL_MMIO_BASE + 0x14))
#define ACCEL_REG_LPDDR5_MMU    ((volatile uint32_t*)(ACCEL_MMIO_BASE + 0x18))

// Synapse Weight memory pointer (64KB SRAM space mapped to 8-bit weights)
#define SYNAPSE_RAM            ((volatile uint8_t*)SYNAPSE_RAM_BASE)

// Global memory buffer to log classification results (located in RISC-V SRAM)
#define LOG_BUFFER_SIZE 16
volatile uint32_t classification_log[LOG_BUFFER_SIZE];
volatile uint32_t log_index = 0;

// Helper function to program weights
void program_synapse(uint8_t pre_neuron, uint8_t post_neuron, uint8_t weight) {
    uint32_t address_offset = ((uint32_t)pre_neuron * 256) + post_neuron;
    SYNAPSE_RAM[address_offset] = weight;
}

// System initialization
void init_hardware(void) {
    // 1. Set Spike threshold to 255
    *NEURO_REG_THRESHOLD = 255;
    
    // 2. Set leak decay rate to 3
    *NEURO_REG_LEAK = 3;
    
    // 3. Clear input and output registers (8 words = 256 bits)
    for (int i = 0; i < 8; i++) {
        NEURO_REG_INPUT_SPIKES[i] = 0;
    }

    // 4. Initialize LPDDR5 Memory Controller (V2 Upgrade)
    *ACCEL_REG_LPDDR5_CTRL = 0x01; // Enable PHY
    while (!((*ACCEL_REG_LPDDR5_CTRL) & 0x04)) {
        // Wait for PHY PLL Lock and Calibration to be Ready (Bit 2)
    }

    // 5. Initialize Virtual Memory Translation (MMU SATP table pointer)
    *ACCEL_REG_LPDDR5_MMU = 0x80000000; // Map Page directory base (0x8000_0000)

    // 6. Enable STDP On-Chip Spiking Learning Engine
    *NEURO_REG_STDP_CTRL = 0x01; // Enable STDP dynamic weight updates
}

// Program classification templates into the Synaptic Memory
void load_neuromorphic_weights(void) {
    // Clear synapse space to 0 (unconnected)
    for (uint32_t i = 0; i < 65536; i++) {
        SYNAPSE_RAM[i] = 0;
    }

    // Pattern A Classifier (Neuron 0: Horizontal Line)
    // Connect Inputs 4, 5, 6, 7 (representing row 1 pixels) to Neuron 0
    program_synapse(4, 0, 75);
    program_synapse(5, 0, 75);
    program_synapse(6, 0, 75);
    program_synapse(7, 0, 75);

    // Pattern B Classifier (Neuron 1: Vertical Line)
    // Connect Inputs 1, 5, 9, 13 (representing col 1 pixels) to Neuron 1
    program_synapse(1, 1, 75);
    program_synapse(5, 1, 75);
    program_synapse(9, 1, 75);
    program_synapse(13, 1, 75);

    // Pattern C Classifier (Neuron 2: Cross Pattern Classifier)
    // Connect inputs representing both Row 1 and Col 1
    program_synapse(1, 2, 45);
    program_synapse(4, 2, 45);
    program_synapse(5, 2, 50);
    program_synapse(6, 2, 45);
    program_synapse(7, 2, 45);
    program_synapse(9, 2, 45);
    program_synapse(13, 2, 45);
}

// Feeds input spike patterns, triggers accelerator and polls for Done
uint32_t run_snn_inference(uint32_t input_mask_word0) {
    // 1. Load active spikes into Input spike register word 0 (inputs 0-31)
    NEURO_REG_INPUT_SPIKES[0] = input_mask_word0;
    
    // 2. Trigger the Neuromorphic Accelerator run (Start bit = [0])
    *NEURO_REG_CTRL = 0x00000001;
    
    // 3. Polling: Wait for the Done flag (Bit 2 of Control Reg) to go high
    // The hardware automatically clears the Start bit once execution begins.
    while (!((*NEURO_REG_CTRL) & 0x00000004)) {
        // Spin waiting for hardware sequencer to finish SNN cycle
    }
    
    // 4. Read output spike result word 0 (neurons 0-31)
    uint32_t output_spikes = NEURO_REG_OUT_SPIKES[0];
    
    return output_spikes;
}

// Main execution entry point (Called by RISC-V startup vector)
int main(void) {
    // Initialize registers
    init_hardware();
    
    // Load Synaptic weights BRAM crossbar
    load_neuromorphic_weights();
    
    // Test patterns vector
    uint32_t test_patterns[4] = {
        0x000000F0, // Pattern A: Row 1 active (inputs 4, 5, 6, 7) -> Expected: Neuron 0 Fires
        0x00002222, // Pattern B: Col 1 active (inputs 1, 5, 9, 13) -> Expected: Neuron 1 Fires
        0x000022F2, // Pattern C: Cross active (inputs 1, 4, 5, 6, 7, 9, 13) -> Expected: Neuron 2 Fires
        0x00000000  // Idle / Distractor
    };

    // Run SNN inference loops
    while (1) {
        for (int p = 0; p < 4; p++) {
            uint32_t result = run_snn_inference(test_patterns[p]);
            
            // Log output classification to SRAM memory buffer
            classification_log[log_index] = result;
            log_index = (log_index + 1) % LOG_BUFFER_SIZE;
            
            // Artificial delay loop before next temporal feed
            for (volatile int delay = 0; delay < 1000; delay++);
        }
    }

    return 0;
}
