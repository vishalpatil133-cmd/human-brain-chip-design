// =============================================================================
// Module: gpu_chiplet
// Description: Synthesizable GPU Chiplet (Die C) containing:
//              1. GPU registers for rendering configurations (colors, vertex coords).
//              2. Parallel execution graphics pixel blender.
//              3. UCIe Slave Interconnect Bridge to receive requests from CPU.
// Architect: Principal Semiconductor Architect
// =============================================================================

module gpu_chiplet (
    input wire clk,
    input wire rst_n,
    
    // UCIe Physical Link Ports (GPU Side)
    input wire [15:0]  tx_data,
    input wire         tx_val,
    output wire        tx_rdy,
    
    output wire [15:0] rx_data,
    output wire        rx_val,
    input wire         rx_rdy
);

    // =========================================================================
    // UCIe Slave Interface Bridge
    // =========================================================================
    wire [31:0] ucie_addr_c;
    wire [31:0] ucie_wdata_c;
    wire        ucie_we_c;
    wire        ucie_req_c;
    
    reg         ucie_ready_c;
    reg  [31:0] ucie_rdata_c;
    reg         ucie_valid_c;

    chiplet_link d2d_bridge (
        .clk(clk),
        .rst_n(rst_n),
        
        // Initiator port not used on GPU side (it acts as target)
        .tx_addr_a(32'd0),
        .tx_wdata_a(32'd0),
        .tx_we_a(1'b0),
        .tx_req_a(1'b0),
        .tx_ready_a(),
        .rx_rdata_a(),
        .rx_valid_a(),
        
        // Connect to GPU registers
        .rx_addr_b(ucie_addr_c),
        .rx_wdata_b(ucie_wdata_c),
        .rx_we_b(ucie_we_c),
        .rx_req_b(ucie_req_c),
        .tx_ready_b(ucie_ready_c),
        .tx_rdata_b(ucie_rdata_c),
        .tx_valid_b(ucie_valid_c),
        
        // Physical Interposer PHY Ports
        .ucie_data_tx(rx_data), // Cross routing
        .ucie_val_tx(rx_val),
        .ucie_rdy_tx(rx_rdy),
        
        .ucie_data_rx(tx_data),
        .ucie_val_rx(tx_val),
        .ucie_rdy_rx(tx_rdy)
    );

    // =========================================================================
    // GPU Execution Registers
    // =========================================================================
    reg [31:0] reg_gpu_ctrl;       // Offset 0x00: [0]=Start Draw, [1]=Ready
    reg [31:0] reg_gpu_color_a;    // Offset 0x10: RGBA color A
    reg [31:0] reg_gpu_color_b;    // Offset 0x14: RGBA color B
    reg [31:0] reg_gpu_alpha;      // Offset 0x18: Alpha scale (0-255)
    wire [31:0] reg_gpu_out_pixel;  // Offset 0x1C: Blended output (Read-Only)
    
    reg [31:0] reg_gpu_vertex_x;   // Offset 0x20: Vertex X coordinate
    reg [31:0] reg_gpu_vertex_y;   // Offset 0x24: Vertex Y coordinate

    // Parallel blend execution: R_out = (R_a * alpha + R_b * (255 - alpha)) / 255
    assign reg_gpu_out_pixel[31:24] = ((reg_gpu_color_a[31:24] * reg_gpu_alpha[7:0]) + (reg_gpu_color_b[31:24] * (8'd255 - reg_gpu_alpha[7:0]))) / 8'd255;
    assign reg_gpu_out_pixel[23:16] = ((reg_gpu_color_a[23:16] * reg_gpu_alpha[7:0]) + (reg_gpu_color_b[23:16] * (8'd255 - reg_gpu_alpha[7:0]))) / 8'd255;
    assign reg_gpu_out_pixel[15:8]  = ((reg_gpu_color_a[15:8]  * reg_gpu_alpha[7:0]) + (reg_gpu_color_b[15:8]  * (8'd255 - reg_gpu_alpha[7:0]))) / 8'd255;
    assign reg_gpu_out_pixel[7:0]   = 8'hFF; // Alpha locked to 1.0

    // MMIO Registers Interface
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_gpu_ctrl     <= 32'h0000_0002; // Default: Ready
            reg_gpu_color_a  <= 32'd0;
            reg_gpu_color_b  <= 32'd0;
            reg_gpu_alpha    <= 32'd128; // 50% blend
            reg_gpu_vertex_x <= 32'd0;
            reg_gpu_vertex_y <= 32'd0;
            ucie_ready_c     <= 1'b1;
            ucie_valid_c     <= 1'b0;
            ucie_rdata_c     <= 32'd0;
        end else begin
            ucie_valid_c <= 1'b0;
            
            // Clear auto-clearing Start bit
            if (reg_gpu_ctrl[0])
                reg_gpu_ctrl[0] <= 1'b0;

            if (ucie_req_c) begin
                ucie_ready_c <= 1'b0;
                
                // Write Register
                if (ucie_we_c) begin
                    case (ucie_addr_c[7:0])
                        8'h00: reg_gpu_ctrl[1:0] <= ucie_wdata_c[1:0];
                        8'h10: reg_gpu_color_a   <= ucie_wdata_c;
                        8'h14: reg_gpu_color_b   <= ucie_wdata_c;
                        8'h18: reg_gpu_alpha     <= ucie_wdata_c;
                        8'h20: reg_gpu_vertex_x  <= ucie_wdata_c;
                        8'h24: reg_gpu_vertex_y  <= ucie_wdata_c;
                    endcase
                end
                
                // Read Register
                case (ucie_addr_c[7:0])
                    8'h00: ucie_rdata_c <= reg_gpu_ctrl;
                    8'h10: ucie_rdata_c <= reg_gpu_color_a;
                    8'h14: ucie_rdata_c <= reg_gpu_color_b;
                    8'h18: ucie_rdata_c <= reg_gpu_alpha;
                    8'h1C: ucie_rdata_c <= reg_gpu_out_pixel;
                    8'h20: ucie_rdata_c <= reg_gpu_vertex_x;
                    8'h24: ucie_rdata_c <= reg_gpu_vertex_y;
                    default: ucie_rdata_c <= 32'd0;
                endcase
                
                ucie_valid_c <= 1'b1;
            end else begin
                ucie_ready_c <= 1'b1;
            end
        end
    end

endmodule
