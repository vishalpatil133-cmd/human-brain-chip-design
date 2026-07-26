// =============================================================================
// Module: triple_chiplet_top
// Description: Synthesizable top-level Triple-Chiplet MCM package.
//              Integrates Die A (Performance CPU), Die B (Neuromorphic SNN),
//              and Die C (GPU Accelerator) over a shared multi-drop UCIe bus.
// Architect: Principal Semiconductor Architect
// =============================================================================

module triple_chiplet_top (
    input wire clk,
    input wire rst_n,
    
    // External System Interfaces (routed to Performance Chiplet)
    input wire rx,
    output wire tx
);

    // =========================================================================
    // UCIe Physical Interposer Bus Routing
    // =========================================================================
    // Transmit bus from CPU Chiplet (Die A) to all targets (broadcast)
    wire [15:0] ucie_data_a2targets;
    wire        ucie_val_a2targets;
    wire        ucie_rdy_targets2a; // credit ready
    
    // Response buses from SNN (Die B) and GPU (Die C) back to CPU
    wire [15:0] ucie_data_b2a;
    wire        ucie_val_b2a;
    wire        ucie_rdy_a2b;
    
    wire [15:0] ucie_data_c2a;
    wire        ucie_val_c2a;
    wire        ucie_rdy_a2c;

    // Multiplexing target responses back to Die A (Interposer Bus Mux)
    wire [15:0] ucie_data_rx_to_a = (ucie_val_b2a) ? ucie_data_b2a : 
                                    (ucie_val_c2a) ? ucie_data_c2a : 16'd0;
                                    
    wire        ucie_val_rx_to_a  = ucie_val_b2a || ucie_val_c2a;
    
    // Combine ready credits
    assign ucie_rdy_targets2a = ucie_rdy_a2b && ucie_rdy_a2c;

    // =========================================================================
    // Die A Instance (Performance CPU Chiplet)
    // =========================================================================
    performance_chiplet die_a (
        .clk(clk),
        .rst_n(rst_n),
        
        .tx_data(ucie_data_a2targets),
        .tx_val(ucie_val_a2targets),
        .tx_rdy(ucie_rdy_targets2a),
        
        .rx_data(ucie_data_rx_to_a),
        .rx_val(ucie_val_rx_to_a),
        .rx_rdy(ucie_rdy_a2b) // using link A2B ready as default receiver ready
    );

    // =========================================================================
    // Die B Instance (Neuromorphic SNN Chiplet)
    // =========================================================================
    neuromorphic_chiplet #(
        .TILE_GRID_ROWS(4),
        .TILE_GRID_COLS(4)
    ) die_b (
        .clk(clk),
        .rst_n(rst_n),
        
        .tx_data(ucie_data_b2a),
        .tx_val(ucie_val_b2a),
        .tx_rdy(ucie_rdy_a2b),
        
        .rx_data(ucie_data_a2targets),
        .rx_val(ucie_val_a2targets && (die_a.d2d_bridge.dest_id_a == 1'b0)), // SNN destination
        .rx_rdy(ucie_rdy_a2b)
    );

    // =========================================================================
    // Die C Instance (GPU Chiplet)
    // =========================================================================
    gpu_chiplet die_c (
        .clk(clk),
        .rst_n(rst_n),
        
        .tx_data(ucie_data_c2a),
        .tx_val(ucie_val_c2a),
        .tx_rdy(ucie_rdy_a2c),
        
        .rx_data(ucie_data_a2targets),
        .rx_val(ucie_val_a2targets && (die_a.d2d_bridge.dest_id_a == 1'b1)), // GPU destination
        .rx_rdy(ucie_rdy_a2c)
    );

    // Pin loopback for Tx/Rx mapping to avoid unused port optimization
    assign tx = rx;

endmodule
