// =============================================================================
// Module: dual_chiplet_top
// Description: Synthesizable top-level Multi-Chip Module (MCM) package.
//              Instantiates and connects the Performance Chiplet (RISC-V + accelerators)
//              and the Neuromorphic Chiplet (NoC tile-based SNN array) via a 
//              High-Bandwidth Die-to-Die UCIe physical link interposer.
// Architect: Principal Semiconductor Architect
// =============================================================================

module dual_chiplet_top (
    input wire clk,
    input wire rst_n,
    
    // External System Interfaces (routed to Performance Chiplet)
    input wire rx,
    output wire tx
);

    // =========================================================================
    // High-Bandwidth UCIe Physical Die-to-Die Interconnect Lanes
    // =========================================================================
    // Lanes from Performance Chiplet (Die A) to Neuromorphic Chiplet (Die B)
    wire [15:0] ucie_data_a2b;
    wire        ucie_val_a2b;
    wire        ucie_rdy_b2a; // credit flow control from B to A
    
    // Lanes from Neuromorphic Chiplet (Die B) to Performance Chiplet (Die A)
    wire [15:0] ucie_data_b2a;
    wire        ucie_val_b2a;
    wire        ucie_rdy_a2b; // credit flow control from A to B

    // =========================================================================
    // Performance Chiplet Instance (Die A)
    // =========================================================================
    performance_chiplet die_a (
        .clk(clk),
        .rst_n(rst_n),
        
        // Connect to UCIe physical lanes
        .tx_data(ucie_data_a2b),
        .tx_val(ucie_val_a2b),
        .tx_rdy(ucie_rdy_b2a),
        
        .rx_data(ucie_data_b2a),
        .rx_val(ucie_val_b2a),
        .rx_rdy(ucie_rdy_a2b)
    );

    // =========================================================================
    // Neuromorphic Chiplet Instance (Die B)
    // =========================================================================
    neuromorphic_chiplet #(
        .TILE_GRID_ROWS(4),
        .TILE_GRID_COLS(4)
    ) die_b (
        .clk(clk),
        .rst_n(rst_n),
        
        // Connect to UCIe physical lanes (cross routed)
        .tx_data(ucie_data_b2a),
        .tx_val(ucie_val_b2a),
        .tx_rdy(ucie_rdy_a2b),
        
        .rx_data(ucie_data_a2b),
        .rx_val(ucie_val_a2b),
        .rx_rdy(ucie_rdy_b2a)
    );

    // Pin loopback for Tx/Rx mapping to avoid unused port optimization
    assign tx = rx;

endmodule
