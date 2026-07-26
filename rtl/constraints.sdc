# ============================================================================
# Synopsys Design Constraints (.sdc) for Triple-Chiplet MCM ASIC Synthesis
# Defines Clock Frequencies, I/O Delays, and Clock Domain Crossing (CDC) False Paths
# ============================================================================

# 1. Define Primary Clocks
create_clock -name clk_die_a -period 10.00 [get_ports clk_die_a]  # 100 MHz (Performance CPU)
create_clock -name clk_die_b -period  5.00 [get_ports clk_die_b]  # 200 MHz (Neuromorphic Brain Core)
create_clock -name clk_die_c -period  6.67 [get_ports clk_die_c]  # 150 MHz (GPU Graphics Blender)

# 2. Clock Uncertainty & Jitter
set_clock_uncertainty 0.20 [get_clocks {clk_die_a clk_die_b clk_die_c}]

# 3. Asynchronous Clock Groups (Clock Domain Crossing - CDC False Paths)
set_clock_groups -asynchronous \
    -group [get_clocks clk_die_a] \
    -group [get_clocks clk_die_b] \
    -group [get_clocks clk_die_c]

# 4. Input & Output Constraints
set_input_delay  -max 2.0 -clock clk_die_a [get_ports rst_n]
set_output_delay -max 2.0 -clock clk_die_b [get_ports die_b_spike_outputs*]
set_output_delay -max 2.0 -clock clk_die_c [get_ports die_c_pixel_out*]

# 5. Operating Conditions & Load
set_driving_cell -lib_cell INVX1 [all_inputs]
set_load 0.05 [all_outputs]
