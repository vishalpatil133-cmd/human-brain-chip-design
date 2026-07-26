// =============================================================================
// Module: chiplet_link
// Description: Synthesizable UCIe-style D2D link controller upgraded to support
//              multi-chiplet routing (Die A Initiator -> Die B SNN vs Die C GPU).
//              Performs packet serialization and destination-ID decoding.
// Architect: Principal Semiconductor Architect
// =============================================================================

module chiplet_link (
    input wire clk,
    input wire rst_n,
    
    // =========================================================================
    // Client Side A: Performance Chiplet Internal Bus
    // =========================================================================
    input wire [31:0] tx_addr_a,
    input wire [31:0] tx_wdata_a,
    input wire        tx_we_a,
    input wire        tx_req_a,
    output reg        tx_ready_a,
    output reg [31:0] rx_rdata_a,
    output reg        rx_valid_a,
    
    // =========================================================================
    // Target Interface: Neuromorphic Chiplet (Dest ID = 0)
    // =========================================================================
    output reg [31:0] rx_addr_b,
    output reg [31:0] rx_wdata_b,
    output reg        rx_we_b,
    output reg        rx_req_b,
    input wire        tx_ready_b,
    input wire [31:0] tx_rdata_b,
    input wire        tx_valid_b,
    
    // =========================================================================
    // Target Interface: GPU Chiplet (Dest ID = 1)
    // =========================================================================
    output reg [31:0] rx_addr_c,
    output reg [31:0] rx_wdata_c,
    output reg        rx_we_c,
    output reg        rx_req_c,
    input wire        tx_ready_c,
    input wire [31:0] tx_rdata_c,
    input wire        tx_valid_c,
    
    // =========================================================================
    // Die-to-Die Interposer PHY Ports
    // =========================================================================
    output reg [15:0] ucie_data_tx,
    output reg        ucie_val_tx,
    input wire        ucie_rdy_tx,
    
    input wire [15:0] ucie_data_rx,
    input wire        ucie_val_rx,
    output reg        ucie_rdy_rx
);

    // Destination ID Decode based on Address
    // Address 0x0001_xxxx - 0x0002_xxxx: SNN Core (Dest ID = 0)
    // Address 0x0003_xxxx: GPU Core (Dest ID = 1)
    wire dest_id_a = (tx_addr_a[31:16] == 16'h0003); // 1 = GPU, 0 = SNN

    // =========================================================================
    // Transmit (TX) Serialization FSM
    // =========================================================================
    localparam TX_IDLE   = 3'd0;
    localparam TX_WORD0  = 3'd1;
    localparam TX_WORD1  = 3'd2;
    localparam TX_WORD2  = 3'd3;
    localparam TX_WORD3  = 3'd4;
    localparam TX_WAIT   = 3'd5;

    reg [2:0] tx_state;
    reg [31:0] reg_addr_a;
    reg [31:0] reg_wdata_a;
    reg        reg_we_a;
    reg        reg_dest_a;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state     <= TX_IDLE;
            tx_ready_a   <= 1'b1;
            ucie_val_tx  <= 1'b0;
            ucie_data_tx <= 16'd0;
            reg_addr_a   <= 32'd0;
            reg_wdata_a  <= 32'd0;
            reg_we_a     <= 1'b0;
            reg_dest_a   <= 1'b0;
        end else begin
            case (tx_state)
                TX_IDLE: begin
                    ucie_val_tx <= 1'b0;
                    if (tx_req_a && tx_ready_a) begin
                        reg_addr_a  <= tx_addr_a;
                        reg_wdata_a <= tx_wdata_a;
                        reg_we_a    <= tx_we_a;
                        reg_dest_a  <= dest_id_a;
                        tx_ready_a  <= 1'b0;
                        tx_state    <= TX_WORD0;
                    end else begin
                        tx_ready_a <= 1'b1;
                    end
                end
                
                TX_WORD0: begin
                    if (ucie_rdy_tx) begin
                        // Packet header: [15]=WE, [14]=Dest ID, [13:0]=Address [29:16]
                        ucie_data_tx <= {reg_we_a, reg_dest_a, reg_addr_a[29:16]};
                        ucie_val_tx  <= 1'b1;
                        tx_state     <= TX_WORD1;
                    end
                end
                
                TX_WORD1: begin
                    if (ucie_rdy_tx) begin
                        ucie_data_tx <= reg_addr_a[15:0];
                        ucie_val_tx  <= 1'b1;
                        if (reg_we_a) begin
                            tx_state <= TX_WORD2;
                        end else begin
                            ucie_val_tx <= 1'b0;
                            tx_state    <= TX_WAIT;
                        end
                    end
                end
                
                TX_WORD2: begin
                    if (ucie_rdy_tx) begin
                        ucie_data_tx <= reg_wdata_a[31:16];
                        ucie_val_tx  <= 1'b1;
                        tx_state     <= TX_WORD3;
                    end
                end
                
                TX_WORD3: begin
                    if (ucie_rdy_tx) begin
                        ucie_data_tx <= reg_wdata_a[15:0];
                        ucie_val_tx  <= 1'b1;
                        tx_state     <= TX_IDLE;
                    end
                end
                
                TX_WAIT: begin
                    ucie_val_tx <= 1'b0;
                    if (rx_valid_a) begin
                        tx_state <= TX_IDLE;
                    end
                end
                
                default: tx_state <= TX_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Receive (RX) Deserialization FSM
    // =========================================================================
    localparam RX_IDLE   = 3'd0;
    localparam RX_WORD0  = 3'd1;
    localparam RX_WORD1  = 3'd2;
    localparam RX_WORD2  = 3'd3;
    localparam RX_WORD3  = 3'd4;
    localparam RX_RESP_B = 3'd5;
    localparam RX_RESP_C = 3'd6;

    reg [2:0] rx_state;
    reg        rx_write_type;
    reg        rx_dest_id;
    reg [31:0] reg_addr_target;
    reg [31:0] reg_wdata_target;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state         <= RX_IDLE;
            ucie_rdy_rx      <= 1'b1;
            reg_addr_target  <= 32'd0;
            reg_wdata_target <= 32'd0;
            rx_we_b          <= 1'b0;
            rx_req_b         <= 1'b0;
            rx_we_c          <= 1'b0;
            rx_req_c         <= 1'b0;
            rx_rdata_a       <= 32'd0;
            rx_valid_a       <= 1'b0;
            rx_write_type    <= 1'b0;
            rx_dest_id       <= 1'b0;
        end else begin
            case (rx_state)
                RX_IDLE: begin
                    rx_req_b   <= 1'b0;
                    rx_req_c   <= 1'b0;
                    rx_valid_a <= 1'b0;
                    ucie_rdy_rx <= 1'b1;
                    if (ucie_val_rx) begin
                        rx_write_type <= ucie_data_rx[15];
                        rx_dest_id    <= ucie_data_rx[14];
                        reg_addr_target[29:16] <= ucie_data_rx[13:0];
                        reg_addr_target[31:30] <= 2'b00;
                        ucie_rdy_rx   <= 1'b0;
                        rx_state      <= RX_WORD1;
                    end
                end
                
                RX_WORD1: begin
                    ucie_rdy_rx <= 1'b1;
                    if (ucie_val_rx) begin
                        reg_addr_target[15:0] <= ucie_data_rx;
                        ucie_rdy_rx           <= 1'b0;
                        if (rx_write_type) begin
                            rx_state <= RX_WORD2;
                        end else begin
                            // Read request decoding
                            if (rx_dest_id == 1'b0) begin
                                rx_we_b  <= 1'b0;
                                rx_req_b <= 1'b1;
                                rx_state <= RX_RESP_B;
                            end else begin
                                rx_we_c  <= 1'b0;
                                rx_req_c <= 1'b1;
                                rx_state <= RX_RESP_C;
                            end
                        end
                    end
                end
                
                RX_WORD2: begin
                    ucie_rdy_rx <= 1'b1;
                    if (ucie_val_rx) begin
                        reg_wdata_target[31:16] <= ucie_data_rx;
                        ucie_rdy_rx             <= 1'b0;
                        rx_state                <= RX_WORD3;
                    end
                end
                
                RX_WORD3: begin
                    ucie_rdy_rx <= 1'b1;
                    if (ucie_val_rx) begin
                        reg_wdata_target[15:0] <= ucie_data_rx;
                        ucie_rdy_rx            <= 1'b0;
                        
                        // Output write request to target
                        if (rx_dest_id == 1'b0) begin
                            rx_addr_b  <= reg_addr_target;
                            rx_wdata_b <= {reg_wdata_target[31:16], ucie_data_rx};
                            rx_we_b    <= 1'b1;
                            rx_req_b   <= 1'b1;
                        end else begin
                            rx_addr_c  <= reg_addr_target;
                            rx_wdata_c <= {reg_wdata_target[31:16], ucie_data_rx};
                            rx_we_c    <= 1'b1;
                            rx_req_c   <= 1'b1;
                        end
                        rx_state <= RX_IDLE;
                    end
                end
                
                RX_RESP_B: begin
                    rx_req_b <= 1'b0;
                    if (tx_ready_b && tx_valid_b) begin
                        rx_rdata_a <= tx_rdata_b;
                        rx_valid_a <= 1'b1;
                        rx_state   <= RX_IDLE;
                    end
                end
                
                RX_RESP_C: begin
                    rx_req_c <= 1'b0;
                    if (tx_ready_c && tx_valid_c) begin
                        rx_rdata_a <= tx_rdata_c;
                        rx_valid_a <= 1'b1;
                        rx_state   <= RX_IDLE;
                    end
                end
                
                default: rx_state <= RX_IDLE;
            endcase
        end
    end

endmodule
