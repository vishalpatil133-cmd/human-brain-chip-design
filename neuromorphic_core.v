//==============================================================================
// Neuromorphic Core RTL
// 256-Neuron Digital LIF Crossbar with 8-bit Synaptic Weights
// Synthesizable Verilog implementation
//
// Architecture:
//   - SRAM-like memory interface for 256x256 synapse weight matrix
//   - 256 parallel LIF neuron units with shared control
//   - Event-driven computation: only active synapses consume energy
//   - Activity counter for dynamic energy estimation
//==============================================================================

`timescale 1ns / 1ps

module neuromorphic_core (
    input  wire         clk,
    input  wire         rst_n,
    
    // Control
    input  wire         enable,
    input  wire         clear_activity,
    
    // Weight Memory Interface (SRAM-like)
    output wire [15:0]  weight_addr,      // 8-bit row + 8-bit column
    input  wire [7:0]   weight_rdata,     // 8-bit signed weight (-128 to 127)
    output wire         weight_req,        // Memory read request
    input  wire         weight_valid,      // Memory read data valid
    
    // Input Spike Interface
    input  wire [255:0] input_spikes,      // 256 binary spike inputs
    input  wire         input_valid,       // Input spike valid
    
    // Output Spike Interface
    output wire [255:0] output_spikes,     // 256 binary spike outputs
    output wire         output_valid,      // Output spike valid
    
    // Energy/Activity Monitoring
    output reg [31:0]   total_spike_count,     // Total accumulated spikes
    output reg [31:0]   active_synapse_count,  // Total synapse activations
    output wire [7:0]   current_activity       // Spikes in current cycle
);

//==============================================================================
// Parameters
//==============================================================================
parameter NUM_NEURONS = 256;
parameter MEMBRANE_WIDTH = 16;
parameter WEIGHT_WIDTH = 8;
parameter LEAK_FACTOR = 10;  // 1/alpha = 1/exp(-dt/tau) approximation

//==============================================================================
// Internal signals
//==============================================================================

// Neuron state memories
reg [MEMBRANE_WIDTH-1:0] membrane_potential [0:NUM_NEURONS-1];
reg [7:0] refractory_counter [0:NUM_NEURONS-1];
reg [NUM_NEURONS-1:0] spike_out_reg;

// FSM states
typedef enum logic [2:0] {
    IDLE,
    WEIGHT_FETCH,
    INTEGRATE,
    LEAK_UPDATE,
    THRESHOLD_CHECK,
    OUTPUT_GEN
} state_t;

state_t state;

// Internal counters
reg [15:0] current_neuron;
reg [15:0] current_synapse;
reg [7:0] active_synapses_this_neuron;
reg [NUM_NEURONS-1:0] input_spikes_reg;

// Accumulation
reg [31:0] sum_accumulator;
reg [15:0] synapse_counter_internal;

// Helper wires
wire [7:0] row_addr = current_neuron[7:0];
wire [7:0] col_addr = current_synapse[7:0];
assign weight_addr = {current_neuron[7:0], current_synapse[7:0]};

assign current_activity = |spike_out_reg ? 8'd1 : 8'd0;

//==============================================================================
// Neuron Parameters (shared across all neurons)
//==============================================================================
localparam [MEMBRANE_WIDTH-1:0] V_THRESHOLD = 16'd1000;  // Normalized threshold
localparam [MEMBRANE_WIDTH-1:0] V_RESET = 16'd0;
localparam [7:0] REFRACTORY_PERIOD = 8'd20;  // Timesteps
localparam [MEMBRANE_WIDTH-1:0] LEAK_SCALE = 16'd950;  // ~exp(-0.1/20.0) * 1000

//==============================================================================
// Activity Counters
//==============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        total_spike_count <= 32'd0;
        active_synapse_count <= 32'd0;
    end else if (clear_activity) begin
        total_spike_count <= 32'd0;
        active_synapse_count <= 32'd0;
    end else if (enable && state == OUTPUT_GEN) begin
        total_spike_count <= total_spike_count + spike_out_reg;
        active_synapse_count <= active_synapse_count + synapse_counter_internal;
    end
end

//==============================================================================
// Main Control FSM
//==============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        current_neuron <= 16'd0;
        current_synapse <= 16'd0;
        sum_accumulator <= 32'd0;
        spike_out_reg <= 256'd0;
        input_spikes_reg <= 256'd0;
        active_synapses_this_neuron <= 8'd0;
        synapse_counter_internal <= 16'd0;
        output_spikes <= 256'd0;
        output_valid <= 1'b0;
        
        // Reset all neurons
        for (int i = 0; i < NUM_NEURONS; i++) begin
            membrane_potential[i] <= V_RESET;
            refractory_counter[i] <= 8'd0;
        end
        
    end else if (enable) begin
        
        case (state)
            
            //==================================================================
            // IDLE: Wait for input spikes
            //==================================================================
            IDLE: begin
                output_valid <= 1'b0;
                if (input_valid) begin
                    input_spikes_reg <= input_spikes;
                    current_neuron <= 16'd0;
                    current_synapse <= 16'd0;
                    sum_accumulator <= 32'd0;
                    spike_out_reg <= 256'd0;
                    synapse_counter_internal <= 16'd0;
                    state <= WEIGHT_FETCH;
                end
            end
            
            //==================================================================
            // WEIGHT_FETCH: Iterate over active synapses for current neuron
            // Event-driven: only process synapses where input spike is active
            //==================================================================
            WEIGHT_FETCH: begin
                if (current_neuron < NUM_NEURONS) begin
                    // Check if this synapse has an active input spike
                    if (input_spikes_reg[current_synapse]) begin
                        // Request weight from memory
                        weight_req <= 1'b1;
                        state <= INTEGRATE;
                    end else begin
                        // Skip inactive synapse
                        if (current_synapse < NUM_NEURONS - 1) begin
                            current_synapse <= current_synapse + 16'd1;
                        end else begin
                            // Finished all synapses for this neuron
                            current_synapse <= 16'd0;
                            state <= LEAK_UPDATE;
                        end
                    end
                end else begin
                    // All neurons processed
                    state <= OUTPUT_GEN;
                end
            end
            
            //==================================================================
            // INTEGRATE: Accumulate weighted input
            //==================================================================
            INTEGRATE: begin
                weight_req <= 1'b0;
                if (weight_valid) begin
                    // Sign-extend the 8-bit signed weight
                    sum_accumulator <= sum_accumulator + {{(MEMBRANE_WIDTH-WEIGHT_WIDTH){weight_rdata[7]}}, weight_rdata};
                    synapse_counter_internal <= synapse_counter_internal + 16'd1;
                    
                    // Move to next synapse
                    if (current_synapse < NUM_NEURONS - 1) begin
                        current_synapse <= current_synapse + 16'd1;
                        state <= WEIGHT_FETCH;
                    end else begin
                        current_synapse <= 16'd0;
                        state <= LEAK_UPDATE;
                    end
                end
            end
            
            //==================================================================
            // LEAK_UPDATE: Apply leaky integration
            // V = V * alpha + I * beta  (approximated with fixed-point)
            //==================================================================
            LEAK_UPDATE: begin
                if (refractory_counter[current_neuron] > 0) begin
                    // In refractory period
                    refractory_counter[current_neuron] <= refractory_counter[current_neuron] - 8'd1;
                    membrane_potential[current_neuron] <= V_RESET;
                end else begin
                    // Leaky integration
                    // V_mem = V_mem * LEAK_SCALE/1000 + input_sum
                    // Approximated as: V_mem = (V_mem * LEAK_SCALE) >> 10 + input_sum
                    membrane_potential[current_neuron] <= 
                        (membrane_potential[current_neuron] * LEAK_SCALE) / 1000 + 
                        sum_accumulator[MEMBRANE_WIDTH-1:0];
                end
                
                sum_accumulator <= 32'd0;
                state <= THRESHOLD_CHECK;
            end
            
            //==================================================================
            // THRESHOLD_CHECK: Detect spike and manage reset
            //==================================================================
            THRESHOLD_CHECK: begin
                if (refractory_counter[current_neuron] == 0 && 
                    membrane_potential[current_neuron] >= V_THRESHOLD) begin
                    // Spike!
                    spike_out_reg[current_neuron] <= 1'b1;
                    membrane_potential[current_neuron] <= V_RESET;
                    refractory_counter[current_neuron] <= REFRACTORY_PERIOD;
                end
                
                // Move to next neuron
                if (current_neuron < NUM_NEURONS - 1) begin
                    current_neuron <= current_neuron + 16'd1;
                    current_synapse <= 16'd0;
                    state <= WEIGHT_FETCH;
                end else begin
                    current_neuron <= 16'd0;
                    state <= OUTPUT_GEN;
                end
            end
            
            //==================================================================
            // OUTPUT_GEN: Drive output spikes and status signals
            //==================================================================
            OUTPUT_GEN: begin
                output_spikes <= spike_out_reg;
                output_valid <= 1'b1;
                state <= IDLE;
            end
            
            default: state <= IDLE;
            
        endcase
    end
end

//==============================================================================
// Debug / Activity Probe
//==============================================================================
reg [31:0] integration_cycles;
reg [31:0] total_cycles;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        integration_cycles <= 32'd0;
        total_cycles <= 32'd0;
    end else if (enable) begin
        total_cycles <= total_cycles + 1;
        if (state == INTEGRATE && weight_valid) begin
            integration_cycles <= integration_cycles + 1;
        end
    end
end

//==============================================================================
// Initialization
//==============================================================================
initial begin
    $display("========================================");
    $display(" Neuromorphic Core (%0d neurons)", NUM_NEURONS);
    $display(" Weight: %0d-bit signed", WEIGHT_WIDTH);
    $display(" Membrane: %0d-bit", MEMBRANE_WIDTH);
    $display(" Threshold: %0d", V_THRESHOLD);
    $display("========================================");
end

endmodule