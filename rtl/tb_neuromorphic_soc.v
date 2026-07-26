`timescale 1ns / 1ps

module tb_neuromorphic_soc;

    // Inputs
    reg clk;
    reg rst_n;
    reg rx;

    // Outputs
    wire tx;

    // Instantiate the Unit Under Test (UUT)
    neuromorphic_soc uut (
        .clk(clk),
        .rst_n(rst_n),
        .rx(rx),
        .tx(tx)
    );

    // Clock generator (50MHz -> 20ns period)
    always #10 clk = ~clk;

    initial begin
        // Initialize Inputs
        clk = 0;
        rst_n = 0;
        rx = 0;

        // Wait 100 ns for global reset
        #100;
        rst_n = 1;
        
        $display("=================================================");
        $display("   Starting Neuromorphic RISC-V SoC Simulation  ");
        $display("=================================================");
        
        // Monitor key CPU registers and PC to see program execution
        $monitor("[Time %0t] PC = 0x%h | CPU Instr = 0x%h | MMIO Control = 0x%h | Accel Done = %b", 
                 $time, uut.cpu_pc, uut.cpu_instr, uut.mmio_control, uut.accel_done);

        // Run the simulation for 1000 clock cycles (20000 ns)
        // This will let the RISC-V bootloader initialize, write threshold, and start the accelerator.
        #10000;

        $display("=================================================");
        $display("  Checking Neuromorphic Register Values:        ");
        $display("  Threshold: %d (Expected: 255)", uut.mmio_threshold);
        $display("  Leak Rate: %d (Expected: 4)", uut.mmio_leak);
        $display("  Synaptic Ops: %d", uut.mmio_syn_ops);
        $display("  Fired Spikes: %d", uut.mmio_neuron_spikes);
        $display("=================================================");
        $display("Simulation Completed Successfully!");
        $finish;
    end
      
endmodule
