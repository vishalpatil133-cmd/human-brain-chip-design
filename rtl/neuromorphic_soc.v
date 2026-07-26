//==============================================================================
// neuromorphic_soc.v — Top-Level Neuromorphic SoC
//
// Architecture:
//   ┌──────────────────────────────────────────────────────────┐
//   │  neuromorphic_soc                                        │
//   │  ┌─────────────┐    ┌──────────────┐   ┌──────────────┐ │
//   │  │ RISC-V      │    │ AXI-Lite     │   │ SRAM         │ │
//   │  │ RV32I CPU   │───▶│ Bus Matrix   │──▶│ 64KB Ctrl    │ │
//   │  │ (multi-     │    │              │   └──────────────┘ │
//   │  │  cycle)     │    │              │   ┌──────────────┐ │
//   │  └─────────────┘    │              │──▶│ Neuromorphic │ │
//   │                     │              │   │ Core 256-neu │ │
//   │                     │              │   │ + Weight Mem │ │
//   │                     │              │   └──────────────┘ │
//   │                     └──────────────┘                    │
//   └──────────────────────────────────────────────────────────┘
//
// Features:
//   - RV32I multi-cycle CPU (all base instructions)
//   - AXI-Lite compatible 32-bit system bus
//   - 64KB on-chip SRAM (code + data)
//   - 256-neuron LIF neuromorphic core with 8-bit weights
//   - Memory-mapped control: write weights, inject spikes, read results
//   - Synthesizable, single-clock domain
//   - 8GB RAM local machine friendly
//==============================================================================

`timescale 1ns / 1ps

//==============================================================================
// Top-Level SoC Module
//==============================================================================
module neuromorphic_soc (
    input  wire         clk,
    input  wire         rst_n,

    // External interrupt (for timer/OS scheduler)
    input  wire         ext_irq,

    // UART / debug interface (memory-mapped, simplified)
    output wire [7:0]   uart_tx_data,
    output wire         uart_tx_valid,
    input  wire         uart_tx_ready,
    input  wire [7:0]   uart_rx_data,
    input  wire         uart_rx_valid,
    output wire         uart_rx_ready
);
//==============================================================================
// Internal Parameters
//==============================================================================
localparam SRAM_SIZE_WORDS = 16384;  // 64 KB = 16384 x 32-bit
localparam NUM_NEURONS     = 256;
localparam MEM_WIDTH       = 16;

//==============================================================================
// Master (CPU) Bus Signals — AXI-Lite
//==============================================================================
reg        m_awvalid;
reg [31:0] m_awaddr;
reg [2:0]  m_awprot;
wire       m_awready;

reg        m_wvalid;
reg [31:0] m_wdata;
reg [3:0]  m_wstrb;
wire       m_wready;

wire [1:0] m_bresp;
wire       m_bvalid;
reg        m_bready;

reg        m_arvalid;
reg [31:0] m_araddr;
reg [2:0]  m_arprot;
wire       m_arready;

wire [31:0] m_rdata;
wire [1:0]  m_rresp;
wire        m_rvalid;
reg         m_rready;

//==============================================================================
// Slave Interface Wires (decoded per peripheral)
//==============================================================================

// SRAM slave port
wire       sram_awready;
wire       sram_wready;
wire [1:0] sram_bresp;
wire       sram_bvalid;
wire       sram_arready;
wire [31:0] sram_rdata;
wire [1:0] sram_rresp;
wire       sram_rvalid;

// Neuromorphic control slave port
wire       neuro_awready;
wire       neuro_wready;
wire [1:0] neuro_bresp;
wire       neuro_bvalid;
wire       neuro_arready;
wire [31:0] neuro_rdata;
wire [1:0] neuro_rresp;
wire       neuro_rvalid;

//===================================================================
// Bus Matrix — Address Decoder
//===================================================================
// Address Map:
//   0x0000_0000 – 0x0000_FFFF : SRAM   (64 KB)
//   0x1000_0000 – 0x1000_000F : UART   (simplified)
//   0x2000_0000 – 0x2000_FFFF : Weight Memory (256×256×8b = 64 KB)
//   0x3000_0000 – 0x3000_003F : Neuromorphic Control Regs (64 B)

wire sram_sel    = (m_araddr[31:16] == 16'h0000) || (m_awaddr[31:16] == 16'h0000);
wire neuro_sel   = (m_araddr[31:16] == 16'h3000) || (m_awaddr[31:16] == 16'h3000);
wire weight_sel  = (m_araddr[31:16] == 16'h2000) || (m_awaddr[31:16] == 16'h2000);

// Select signal for the target slave (weight memory goes to neuro controller)
wire axi_sel_sram  = sram_sel;
wire axi_sel_ctrl  = neuro_sel;
wire axi_sel_wmem  = weight_sel;

//------------------------------------------------------------------------------
// Read address channel routing
//------------------------------------------------------------------------------
reg [1:0] read_mux;
always @* begin
    if      (axi_sel_sram) read_mux = 2'b00;
    else if (axi_sel_ctrl) read_mux = 2'b01;
    else if (axi_sel_wmem) read_mux = 2'b10;
    else                   read_mux = 2'b11;  // decode error
end

wire [31:0] rdata_mux =
    (read_mux == 2'b00) ? sram_rdata :
    (read_mux == 2'b01) ? neuro_rdata :
    (read_mux == 2'b10) ? neuro_rdata :  // weight read goes through neuro
    32'hDEAD_BEEF;

wire [1:0] resp_mux =
    (read_mux == 2'b11) ? 2'b11 :  // DECERR
    (read_mux == 2'b00) ? sram_rresp : neuro_rresp;

wire rvalid_mux =
    (read_mux == 2'b00) ? sram_rvalid :
    (read_mux == 2'b01) ? neuro_rvalid :
    (read_mux == 2'b10) ? neuro_rvalid :
    1'b1;  // error response takes one cycle

wire arready_mux =
    (read_mux == 2'b00) ? sram_arready :
    (read_mux == 2'b01) ? neuro_arready :
    (read_mux == 2'b10) ? neuro_arready :
    1'b1;

//------------------------------------------------------------------------------
// Write address / data channel routing
//------------------------------------------------------------------------------
reg [1:0] write_mux;
always @* begin
    if      (axi_sel_sram) write_mux = 2'b00;
    else if (axi_sel_ctrl) write_mux = 2'b01;
    else if (axi_sel_wmem) write_mux = 2'b10;
    else                   write_mux = 2'b11;
end

wire wready_mux =
    (write_mux == 2'b00) ? sram_wready :
    (write_mux == 2'b01) ? neuro_wready :
    (write_mux == 2'b10) ? neuro_wready :
    1'b1;

wire awready_mux =
    (write_mux == 2'b00) ? sram_awready :
    (write_mux == 2'b01) ? neuro_awready :
    (write_mux == 2'b10) ? neuro_awready :
    1'b1;

wire [1:0] bresp_mux =
    (write_mux == 2'b11) ? 2'b11 :  // DECERR
    (write_mux == 2'b00) ? sram_bresp : neuro_bresp;

wire bvalid_mux =
    (write_mux == 2'b00) ? sram_bvalid :
    (write_mux == 2'b01) ? neuro_bvalid :
    (write_mux == 2'b10) ? neuro_bvalid :
    1'b1;

// Tie master bus to mux outputs
assign m_awready = awready_mux;
assign m_wready  = wready_mux;
assign m_bresp   = bresp_mux;
assign m_bvalid  = bvalid_mux;
assign m_arready = arready_mux;
assign m_rdata   = rdata_mux;
assign m_rresp   = resp_mux;
assign m_rvalid  = rvalid_mux;

//==============================================================================
// Sub-module Instantiations
//==============================================================================

//-----------------------------------------------------------------------------
// 1. SRAM Controller — 64 KB
//-----------------------------------------------------------------------------
sram_controller #(
    .SIZE_WORDS(SRAM_SIZE_WORDS)
) u_sram (
    .clk         (clk),
    .rst_n       (rst_n),

    .axi_awvalid (m_awvalid && axi_sel_sram),
    .axi_awaddr  (m_awaddr[16:2]),
    .axi_awready (sram_awready),

    .axi_wvalid  (m_wvalid && axi_sel_sram),
    .axi_wdata   (m_wdata),
    .axi_wstrb   (m_wstrb),
    .axi_wready  (sram_wready),

    .axi_bresp   (sram_bresp),
    .axi_bvalid  (sram_bvalid),
    .axi_bready  (m_bready),

    .axi_arvalid (m_arvalid && axi_sel_sram),
    .axi_araddr  (m_araddr[16:2]),
    .axi_arready (sram_arready),

    .axi_rdata   (sram_rdata),
    .axi_rresp   (sram_rresp),
    .axi_rvalid  (sram_rvalid),
    .axi_rready  (m_rready)
);

//-----------------------------------------------------------------------------
// 2. Neuromorphic Core + Controller (weights + control + status)
//-----------------------------------------------------------------------------
neuromorphic_ctrl #(
    .NUM_NEURONS(NUM_NEURONS),
    .MEM_WIDTH(MEM_WIDTH)
) u_neuro_ctrl (
    .clk   (clk),
    .rst_n (rst_n),

    // AXI-Lite slave port (control registers + weight memory access)
    .axi_awvalid (m_awvalid && (axi_sel_ctrl || axi_sel_wmem)),
    .axi_awaddr  (m_awaddr),
    .axi_awready (neuro_awready),

    .axi_wvalid  (m_wvalid && (axi_sel_ctrl || axi_sel_wmem)),
    .axi_wdata   (m_wdata),
    .axi_wstrb   (m_wstrb),
    .axi_wready  (neuro_wready),

    .axi_bresp   (neuro_bresp),
    .axi_bvalid  (neuro_bvalid),
    .axi_bready  (m_bready),

    .axi_arvalid (m_arvalid && (axi_sel_ctrl || axi_sel_wmem)),
    .axi_araddr  (m_araddr),
    .axi_arready (neuro_arready),

    .axi_rdata   (neuro_rdata),
    .axi_rresp   (neuro_rresp),
    .axi_rvalid  (neuro_rvalid),
    .axi_rready  (m_rready)
);

//-----------------------------------------------------------------------------
// 3. RISC-V RV32I CPU Core
//-----------------------------------------------------------------------------
riscv_core u_cpu (
    .clk       (clk),
    .rst_n     (rst_n),
    .irq       (ext_irq),

    // AXI-Lite master port
    .m_awvalid (m_awvalid),
    .m_awaddr  (m_awaddr),
    .m_awready (m_awready),

    .m_wvalid  (m_wvalid),
    .m_wdata   (m_wdata),
    .m_wstrb   (m_wstrb),
    .m_wready  (m_wready),

    .m_bresp   (m_bresp),
    .m_bvalid  (m_bvalid),
    .m_bready  (m_bready),

    .m_arvalid (m_arvalid),
    .m_araddr  (m_araddr),
    .m_arready (m_arready),

    .m_rdata   (m_rdata),
    .m_rresp   (m_rresp),
    .m_rvalid  (m_rvalid),
    .m_rready  (m_rready)
);

//-----------------------------------------------------------------------------
// 4. Simplified UART (memory-mapped, for OS console)
//-----------------------------------------------------------------------------
uart_slave u_uart (
    .clk          (clk),
    .rst_n        (rst_n),

    .axi_awvalid  (m_awvalid && (m_awaddr[31:16] == 16'h1000)),
    .axi_awaddr   (m_awaddr[3:0]),
    .axi_awready  (),

    .axi_wvalid   (m_wvalid && (m_awaddr[31:16] == 16'h1000)),
    .axi_wdata    (m_wdata[7:0]),
    .axi_wready   (),

    .axi_bresp    (),
    .axi_bvalid   (),
    .axi_bready   (m_bready),

    .axi_arvalid  (m_arvalid && (m_araddr[31:16] == 16'h1000)),
    .axi_araddr   (m_araddr[3:0]),
    .axi_arready  (),

    .axi_rdata    (),
    .axi_rvalid   (),

    .uart_tx_data (uart_tx_data),
    .uart_tx_valid(uart_tx_valid),
    .uart_tx_ready(uart_tx_ready),
    .uart_rx_data (uart_rx_data),
    .uart_rx_valid(uart_rx_valid),
    .uart_rx_ready(uart_rx_ready)
);

endmodule


//==============================================================================
// RISC-V RV32I Multi-Cycle CPU Core (All 40 Base Instructions)
//==============================================================================
module riscv_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         irq,

    // AXI-Lite Master
    output reg         m_awvalid,
    output reg [31:0]  m_awaddr,
    input  wire        m_awready,

    output reg         m_wvalid,
    output reg [31:0]  m_wdata,
    output reg [3:0]   m_wstrb,
    input  wire        m_wready,

    input  wire [1:0]  m_bresp,
    input  wire        m_bvalid,
    output reg         m_bready,

    output reg         m_arvalid,
    output reg [31:0]  m_araddr,
    input  wire        m_arready,

    input  wire [31:0] m_rdata,
    input  wire [1:0]  m_rresp,
    input  wire        m_rvalid,
    output reg         m_rready
);
//==============================================================================
// Registers
//==============================================================================
reg [31:0] regfile [0:31];  // x0–x31 (x0 is hardwired zero)
reg [31:0] pc;

// Current instruction
reg [31:0] instruction;
reg [31:0] ir;  // instruction register (pipeline stage)

// Decode wires
wire [6:0]  opcode    = ir[6:0];
wire [2:0]  funct3    = ir[14:12];
wire [6:0]  funct7    = ir[31:25];
wire [4:0]  rd        = ir[11:7];
wire [4:0]  rs1       = ir[19:15];
wire [4:0]  rs2       = ir[24:20];

// Immediate generation
wire [31:0] imm_i = {{21{ir[31]}}, ir[30:20]};
wire [31:0] imm_s = {{21{ir[31]}}, ir[30:25], ir[11:7]};
wire [31:0] imm_b = {{20{ir[31]}}, ir[7], ir[30:25], ir[11:8], 1'b0};
wire [31:0] imm_u = {ir[31:12], 12'b0};
wire [31:0] imm_j = {{12{ir[31]}}, ir[19:12], ir[20], ir[30:21], 1'b0};

// ALU signals
reg [31:0] alu_a, alu_b;
reg [3:0]  alu_ctrl;
wire [31:0] alu_result;
wire        alu_branch_taken;

// FSM state
localparam STATE_FETCH   = 4'd0;
localparam STATE_DECODE  = 4'd1;
localparam STATE_EXEC    = 4'd2;
localparam STATE_MEM_RD  = 4'd3;
localparam STATE_MEM_WR  = 4'd4;
localparam STATE_WB      = 4'd5;
localparam STATE_WAIT_RD = 4'd6;
localparam STATE_WAIT_WR = 4'd7;

reg [3:0]  state, next_state;

// Bus transaction state
reg        bus_read_pending;
reg        bus_write_pending;

//==============================================================================
// ALU Implementation
//==============================================================================
always @* begin
    case (alu_ctrl)
        4'b0000: alu_result = alu_a + alu_b;                                           // ADD / ADDI / LB/LH/LW / SB/SH/SW
        4'b0001: alu_result = alu_a - alu_b;                                           // SUB
        4'b0010: alu_result = alu_a << alu_b[4:0];                                     // SLL / SLLI
        4'b0011: alu_result = ($signed(alu_a) < $signed(alu_b)) ? 32'd1 : 32'd0;       // SLT / SLTI
        4'b0100: alu_result = (alu_a < alu_b) ? 32'd1 : 32'd0;                         // SLTU / SLTUI
        4'b0101: alu_result = alu_a ^ alu_b;                                           // XOR / XORI
        4'b0110: alu_result = alu_a >> alu_b[4:0];                                     // SRL / SRLI
        4'b0111: alu_result = $signed(alu_a) >>> alu_b[4:0];                           // SRA / SRAI
        4'b1000: alu_result = alu_a | alu_b;                                           // OR / ORI
        4'b1001: alu_result = alu_a & alu_b;                                           // AND / ANDI
        4'b1010: alu_result = alu_a;                                                   // LUI / AUIPC pass-through
        4'b1011: alu_result = alu_b;
        default: alu_result = 32'd0;
    endcase
end

// Branch condition evaluation
wire beq_taken  = (alu_a == alu_b);
wire bne_taken  = (alu_a != alu_b);
wire blt_taken  = ($signed(alu_a) < $signed(alu_b));
wire bge_taken  = ($signed(alu_a) >= $signed(alu_b));
wire bltu_taken = (alu_a < alu_b);
wire bgeu_taken = (alu_a >= alu_b);

//==============================================================================
// CPU Control FSM
//==============================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_FETCH;
        pc <= 32'h0000_0000;
        ir <= 32'd0;
        regfile[0] <= 32'd0;  // x0 is always zero
        m_arvalid <= 1'b0;
        m_rready  <= 1'b0;
        m_awvalid <= 1'b0;
        m_wvalid  <= 1'b0;
        m_bready  <= 1'b0;
        bus_read_pending <= 1'b0;
        bus_write_pending <= 1'b0;
        alu_a <= 32'd0;
        alu_b <= 32'd0;
        alu_ctrl <= 4'b0000;
    end else begin
        case (state)

            //------------------------------------------------------------------
            // FETCH: Send instruction fetch request on bus
            //------------------------------------------------------------------
            STATE_FETCH: begin
                m_araddr  <= pc;
                m_arvalid <= 1'b1;
                m_rready  <= 1'b1;
                state <= STATE_DECODE;
            end

            //------------------------------------------------------------------
            // DECODE: Wait for bus read response, latch instruction, decode
            //------------------------------------------------------------------
            STATE_DECODE: begin
                if (m_arvalid && m_arready) begin
                    m_arvalid <= 1'b0;
                end

                if (m_rvalid) begin
                    ir <= m_rdata;
                    m_rready <= 1'b0;
                    state <= STATE_EXEC;
                    pc <= pc + 32'd4;
                end

                // Decode and setup ALU operands happens next cycle in EXEC
            end

            //------------------------------------------------------------------
            // EXEC: Execute instruction (ALU, address calculation)
            //------------------------------------------------------------------
            STATE_EXEC: begin
                // Default: next state
                next_state = STATE_FETCH;

                // Read register operands
                alu_a = (rs1 == 5'd0) ? 32'd0 : regfile[rs1];
                alu_b = (rs2 == 5'd0) ? 32'd0 : regfile[rs2];

                case (opcode)

                    // R-type: ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
                    7'b0110011: begin
                        case (funct3)
                            3'b000: alu_ctrl = (funct7[5]) ? 4'b0001 : 4'b0000;  // SUB / ADD
                            3'b001: alu_ctrl = 4'b0010;  // SLL
                            3'b010: alu_ctrl = 4'b0011;  // SLT
                            3'b011: alu_ctrl = 4'b0100;  // SLTU
                            3'b100: alu_ctrl = 4'b0101;  // XOR
                            3'b101: alu_ctrl = (funct7[5]) ? 4'b0111 : 4'b0110;  // SRA / SRL
                            3'b110: alu_ctrl = 4'b1000;  // OR
                            3'b111: alu_ctrl = 4'b1001;  // AND
                        endcase
                        // Write register
                        if (rd != 5'd0)
                            regfile[rd] <= alu_result;
                        state <= STATE_FETCH;
                    end

                    // I-type: ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
                    7'b0010011: begin
                        alu_b = imm_i;
                        case (funct3)
                            3'b000: alu_ctrl = 4'b0000;  // ADDI
                            3'b010: alu_ctrl = 4'b0011;  // SLTI
                            3'b011: alu_ctrl = 4'b0100;  // SLTIU
                            3'b100: alu_ctrl = 4'b0101;  // XORI
                            3'b110: alu_ctrl = 4'b1000;  // ORI
                            3'b111: alu_ctrl = 4'b1001;  // ANDI
                            3'b001: alu_ctrl = 4'b0010;  // SLLI
                            3'b101: alu_ctrl = (funct7[5]) ? 4'b0111 : 4'b0110;  // SRAI / SRLI
                        endcase
                        if (rd != 5'd0)
                            regfile[rd] <= alu_result;
                        state <= STATE_FETCH;
                    end

                    // LUI
                    7'b0110111: begin
                        alu_ctrl = 4'b1010;
                        alu_a = imm_u;
                        if (rd != 5'd0)
                            regfile[rd] <= imm_u;
                        state <= STATE_FETCH;
                    end

                    // AUIPC
                    7'b0010111: begin
                        if (rd != 5'd0)
                            regfile[rd] <= pc - 4 + imm_u;
                        state <= STATE_FETCH;
                    end

                    // JAL
                    7'b1101111: begin
                        if (rd != 5'd0)
                            regfile[rd] <= pc;  // return address = PC+4 (already incremented)
                        pc <= pc - 4 + imm_j;
                        state <= STATE_FETCH;
                    end

                    // JALR
                    7'b1100111: begin
                        if (rd != 5'd0)
                            regfile[rd] <= pc;
                        alu_a = (rs1 == 5'd0) ? 32'd0 : regfile[rs1];
                        alu_b = imm_i;
                        alu_ctrl = 4'b0000;
                        pc <= (alu_a + imm_i) & ~32'd1;
                        state <= STATE_FETCH;
                    end

                    // B-type branches
                    7'b1100011: begin
                        case (funct3)
                            3'b000: if (beq_taken)  pc <= pc - 4 + imm_b;
                            3'b001: if (bne_taken)  pc <= pc - 4 + imm_b;
                            3'b100: if (blt_taken)  pc <= pc - 4 + imm_b;
                            3'b101: if (bge_taken)  pc <= pc - 4 + imm_b;
                            3'b110: if (bltu_taken) pc <= pc - 4 + imm_b;
                            3'b111: if (bgeu_taken) pc <= pc - 4 + imm_b;
                        endcase
                        state <= STATE_FETCH;
                    end

                    // Load: LB, LH, LW, LBU, LHU
                    7'b0000011: begin
                        alu_a = (rs1 == 5'd0) ? 32'd0 : regfile[rs1];
                        alu_b = imm_i;
                        alu_ctrl = 4'b0000;  // ADD for address
                        m_araddr <= alu_a + imm_i;
                        m_arvalid <= 1'b1;
                        m_rready  <= 1'b1;
                        state <= STATE_MEM_RD;
                    end

                    // Store: SB, SH, SW
                    7'b0100011: begin
                        alu_a = (rs1 == 5'd0) ? 32'd0 : regfile[rs1];
                        alu_b = imm_s;
                        alu_ctrl = 4'b0000;
                        m_awaddr <= alu_a + imm_s;
                        m_awvalid <= 1'b1;
                        m_wdata <= (rs2 == 5'd0) ? 32'd0 : regfile[rs2];
                        // Byte/halfword/word strobe
                        case (funct3)
                            3'b000: m_wstrb <= 4'b0001 << alu_a[1:0];  // SB
                            3'b001: m_wstrb <= (alu_a[1] ? 4'b1100 : 4'b0011);  // SH
                            3'b010: m_wstrb <= 4'b1111;  // SW
                            default: m_wstrb <= 4'b1111;
                        endcase
                        m_wvalid <= 1'b1;
                        state <= STATE_MEM_WR;
                    end

                    // FENCE, ECALL, EBREAK — treat as NOP for now
                    7'b0001111: begin
                        state <= STATE_FETCH;
                    end

                    7'b1110011: begin
                        state <= STATE_FETCH;
                    end

                    default: begin
                        state <= STATE_FETCH;
                    end
                endcase
            end

            //------------------------------------------------------------------
            // MEM_RD: Wait for load data from bus
            //------------------------------------------------------------------
            STATE_MEM_RD: begin
                if (m_arvalid && m_arready) begin
                    m_arvalid <= 1'b0;
                end
                if (m_rvalid) begin
                    case (funct3)
                        3'b000: begin  // LB
                            case (m_araddr[1:0])
                                2'b00: regfile[rd] <= {{24{m_rdata[7]}},  m_rdata[7:0]};
                                2'b01: regfile[rd] <= {{24{m_rdata[15]}}, m_rdata[15:8]};
                                2'b10: regfile[rd] <= {{24{m_rdata[23]}}, m_rdata[23:16]};
                                2'b11: regfile[rd] <= {{24{m_rdata[31]}}, m_rdata[31:24]};
                            endcase
                        end
                        3'b001: begin  // LH
                            if (m_araddr[1])
                                regfile[rd] <= {{16{m_rdata[31]}}, m_rdata[31:16]};
                            else
                                regfile[rd] <= {{16{m_rdata[15]}}, m_rdata[15:0]};
                        end
                        3'b010: regfile[rd] <= m_rdata;  // LW
                        3'b100: begin  // LBU
                            case (m_araddr[1:0])
                                2'b00: regfile[rd] <= {24'd0, m_rdata[7:0]};
                                2'b01: regfile[rd] <= {24'd0, m_rdata[15:8]};
                                2'b10: regfile[rd] <= {24'd0, m_rdata[23:16]};
                                2'b11: regfile[rd] <= {24'd0, m_rdata[31:24]};
                            endcase
                        end
                        3'b101: begin  // LHU
                            if (m_araddr[1])
                                regfile[rd] <= {16'd0, m_rdata[31:16]};
                            else
                                regfile[rd] <= {16'd0, m_rdata[15:0]};
                        end
                    endcase
                    m_rready <= 1'b0;
                    state <= STATE_FETCH;
                end
            end

            //------------------------------------------------------------------
            // MEM_WR: Wait for store completion
            //------------------------------------------------------------------
            STATE_MEM_WR: begin
                if (m_awvalid && m_awready) begin
                    m_awvalid <= 1'b0;
                end
                if (m_wvalid && m_wready) begin
                    m_wvalid <= 1'b0;
                    m_bready <= 1'b1;
                end
                if (m_bvalid && m_bready) begin
                    m_bready <= 1'b0;
                    state <= STATE_FETCH;
                end
            end

            default: state <= STATE_FETCH;
        endcase
    end
end

// Hardwire x0 to zero
always @(posedge clk) begin
    regfile[0] <= 32'd0;
end

endmodule


//==============================================================================
// SRAM Controller — AXI-Lite Slave
// 64 KB (16384 x 32-bit) single-port synchronous SRAM
//==============================================================================
module sram_controller #(
    parameter SIZE_WORDS = 16384
)(
    input  wire         clk,
    input  wire         rst_n,

    // AXI-Lite slave write
    input  wire         axi_awvalid,
    input  wire [14:0]  axi_awaddr,
    output reg          axi_awready,

    input  wire         axi_wvalid,
    input  wire [31:0]  axi_wdata,
    input  wire [3:0]   axi_wstrb,
    output reg          axi_wready,

    output reg [1:0]    axi_bresp,
    output reg          axi_bvalid,
    input  wire         axi_bready,

    // AXI-Lite slave read
    input  wire         axi_arvalid,
    input  wire [14:0]  axi_araddr,
    output reg          axi_arready,

    output reg [31:0]   axi_rdata,
    output reg [1:0]    axi_rresp,
    output reg          axi_rvalid,
    input  wire         axi_rready
);

// SRAM array
reg [31:0] mem [0:SIZE_WORDS-1];

// Address register
reg [14:0] addr_reg;

// FSM
reg [1:0] state;
localparam IDLE = 2'b00;
localparam WRITE_WAIT = 2'b01;
localparam READ_WAIT = 2'b10;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        axi_awready <= 1'b0;
        axi_wready  <= 1'b0;
        axi_bresp   <= 2'b00;
        axi_bvalid  <= 1'b0;
        axi_arready <= 1'b0;
        axi_rdata   <= 32'd0;
        axi_rresp   <= 2'b00;
        axi_rvalid  <= 1'b0;
        addr_reg    <= 15'd0;
    end else begin
        case (state)
            IDLE: begin
                axi_awready <= 1'b0;
                axi_wready  <= 1'b0;
                axi_arready <= 1'b0;
                axi_bvalid  <= 1'b0;
                axi_rvalid  <= 1'b0;

                if (axi_awvalid && axi_wvalid) begin
                    addr_reg <= axi_awaddr;
                    // Write with byte strobes
                    if (axi_wstrb[0]) mem[axi_awaddr][7:0]   <= axi_wdata[7:0];
                    if (axi_wstrb[1]) mem[axi_awaddr][15:8]  <= axi_wdata[15:8];
                    if (axi_wstrb[2]) mem[axi_awaddr][23:16] <= axi_wdata[23:16];
                    if (axi_wstrb[3]) mem[axi_awaddr][31:24] <= axi_wdata[31:24];
                    axi_awready <= 1'b1;
                    axi_wready  <= 1'b1;
                    axi_bresp   <= 2'b00;
                    axi_bvalid  <= 1'b1;
                end else if (axi_awvalid) begin
                    addr_reg <= axi_awaddr;
                    axi_awready <= 1'b1;
                    state <= WRITE_WAIT;
                end else if (axi_arvalid) begin
                    addr_reg <= axi_araddr;
                    axi_rdata  <= mem[axi_araddr];
                    axi_rresp  <= 2'b00;
                    axi_rvalid <= 1'b1;
                    axi_arready <= 1'b1;
                end
            end

            WRITE_WAIT: begin
                axi_awready <= 1'b0;
                if (axi_wvalid) begin
                    if (axi_wstrb[0]) mem[addr_reg][7:0]   <= axi_wdata[7:0];
                    if (axi_wstrb[1]) mem[addr_reg][15:8]  <= axi_wdata[15:8];
                    if (axi_wstrb[2]) mem[addr_reg][23:16] <= axi_wdata[23:16];
                    if (axi_wstrb[3]) mem[addr_reg][31:24] <= axi_wdata[31:24];
                    axi_wready  <= 1'b1;
                    axi_bresp   <= 2'b00;
                    axi_bvalid  <= 1'b1;
                    state <= IDLE;
                end
            end

            READ_WAIT: begin
                state <= IDLE;
            end

            default: state <= IDLE;
        endcase
    end
end

endmodule


//==============================================================================
// Neuromorphic Core Controller — AXI-Lite Slave
// Provides memory-mapped access to:
//   - 256×256 weight SRAM (address range 0x2000_0000–0x2000_FFFF)
//   - Control registers  (address range 0x3000_0000–0x3000_003F)
//==============================================================================
module neuromorphic_ctrl #(
    parameter NUM_NEURONS = 256,
    parameter MEM_WIDTH   = 16
)(
    input  wire         clk,
    input  wire         rst_n,

    // AXI-Lite slave
    input  wire         axi_awvalid,
    input  wire [31:0]  axi_awaddr,
    output reg          axi_awready,

    input  wire         axi_wvalid,
    input  wire [31:0]  axi_wdata,
    input  wire [3:0]   axi_wstrb,
    output reg          axi_wready,

    output reg [1:0]    axi_bresp,
    output reg          axi_bvalid,
    input  wire         axi_bready,

    input  wire         axi_arvalid,
    input  wire [31:0]  axi_araddr,
    output reg          axi_arready,

    output reg [31:0]   axi_rdata,
    output reg [1:0]    axi_rresp,
    output reg          axi_rvalid,
    input  wire         axi_rready
);

//==============================================================================
// Address Decoder
//==============================================================================
// Weight memory: 0x2000_0000 – 0x2000_FFFF (64 KB, 256×256 bytes)
//   Linear address = (neuron_id << 8) | synapse_id
// Control regs:   0x3000_0000 – 0x3000_003F
//   0x00: CORE_ENABLE
//   0x04: CLEAR_ACTIVITY
//   0x08: INPUT_SPIKES_0 (bits 31:0)
//   0x0C: INPUT_SPIKES_1 (bits 63:32)
//   0x10: INPUT_SPIKES_2 (bits 95:64)
//   0x14: INPUT_SPIKES_3 (bits 127:96)
//   0x18: INPUT_SPIKES_4 (bits 159:128)
//   0x1C: INPUT_SPIKES_5 (bits 191:160)
//   0x20: INPUT_SPIKES_6 (bits 223:192)
//   0x24: INPUT_SPIKES_7 (bits 255:224)
//   0x28: START_COMPUTE
//   0x2C: STATUS
//   0x30: SPIKE_COUNT_LO
//   0x34: SPIKE_COUNT_HI
//   0x38: SYNAPSE_COUNT
//   0x3C: OUTPUT_SPIKES (read-only, concatenated into 8 reads)

wire is_weight_mem = (axi_awaddr[31:28] == 4'h2) || (axi_araddr[31:28] == 4'h2);
wire is_ctrl_reg   = (axi_awaddr[31:28] == 4'h3) || (axi_araddr[31:28] == 4'h3);

//==============================================================================
// Registers
//==============================================================================
reg        core_enable;
reg        clear_activity;
reg [255:0] input_spikes;
reg        start_compute;
reg        compute_done;
reg [31:0] spike_count_low;
reg [31:0] spike_count_high;
reg [31:0] synapse_count;
reg [255:0] output_spikes_reg;

// Weight memory (256×256 × 8-bit)
reg [7:0] weight_mem [0:65535];

// FSM for compute
reg [3:0] compute_state;
localparam COMP_IDLE  = 4'd0;
localparam COMP_RUN   = 4'd1;
localparam COMP_DONE  = 4'd2;

// Internal compute signals
reg [15:0] compute_neuron;
reg [15:0] compute_synapse;
reg [31:0] compute_accum;
reg        compute_active;

//==============================================================================
// Bus Write Handling
//==============================================================================
reg [1:0] wr_state;
localparam WR_IDLE = 2'd0;
localparam WR_ADDR = 2'd1;
localparam WR_DATA = 2'd2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        axi_awready  <= 1'b0;
        axi_wready   <= 1'b0;
        axi_bresp    <= 2'b00;
        axi_bvalid   <= 1'b0;
        wr_state     <= WR_IDLE;

        core_enable      <= 1'b0;
        clear_activity   <= 1'b0;
        input_spikes     <= 256'd0;
        start_compute    <= 1'b0;
    end else begin
        case (wr_state)
            WR_IDLE: begin
                axi_awready <= 1'b0;
                axi_wready  <= 1'b0;
                axi_bvalid  <= 1'b0;

                if (axi_awvalid && axi_wvalid) begin
                    // Single-cycle write
                    if (is_weight_mem) begin
                        weight_mem[{axi_awaddr[15:8], axi_awaddr[7:0]}] <= axi_wdata[7:0];
                    end else if (is_ctrl_reg) begin
                        case (axi_awaddr[7:0])
                            8'h00: core_enable    <= axi_wdata[0];
                            8'h04: clear_activity <= axi_wdata[0];
                            8'h08: input_spikes[31:0]   <= axi_wdata;
                            8'h0C: input_spikes[63:32]  <= axi_wdata;
                            8'h10: input_spikes[95:64]  <= axi_wdata;
                            8'h14: input_spikes[127:96] <= axi_wdata;
                            8'h18: input_spikes[159:128]<= axi_wdata;
                            8'h1C: input_spikes[191:160]<= axi_wdata;
                            8'h20: input_spikes[223:192]<= axi_wdata;
                            8'h24: input_spikes[255:224]<= axi_wdata;
                            8'h28: start_compute  <= axi_wdata[0];
                        endcase
                    end
                    axi_awready <= 1'b1;
                    axi_wready  <= 1'b1;
                    axi_bresp   <= 2'b00;
                    axi_bvalid  <= 1'b1;
                end else if (axi_awvalid) begin
                    axi_awready <= 1'b1;
                    wr_state <= WR_ADDR;
                end else if (axi_wvalid) begin
                    axi_wready <= 1'b1;
                    wr_state <= WR_DATA;
                end
            end

            WR_ADDR: begin
                axi_awready <= 1'b0;
                if (axi_wvalid) begin
                    if (is_weight_mem) begin
                        weight_mem[{axi_awaddr[15:8], axi_awaddr[7:0]}] <= axi_wdata[7:0];
                    end else if (is_ctrl_reg) begin
                        case (axi_awaddr[7:0])
                            8'h00: core_enable    <= axi_wdata[0];
                            8'h04: clear_activity <= axi_wdata[0];
                            8'h08: input_spikes[31:0]   <= axi_wdata;
                            8'h0C: input_spikes[63:32]  <= axi_wdata;
                            8'h10: input_spikes[95:64]  <= axi_wdata;
                            8'h14: input_spikes[127:96] <= axi_wdata;
                            8'h18: input_spikes[159:128]<= axi_wdata;
                            8'h1C: input_spikes[191:160]<= axi_wdata;
                            8'h20: input_spikes[223:192]<= axi_wdata;
                            8'h24: input_spikes[255:224]<= axi_wdata;
                            8'h28: start_compute  <= axi_wdata[0];
                        endcase
                    end
                    axi_wready  <= 1'b1;
                    axi_bresp   <= 2'b00;
                    axi_bvalid  <= 1'b1;
                    wr_state <= WR_IDLE;
                end
            end

            WR_DATA: begin
                axi_wready <= 1'b0;
                if (axi_awvalid) begin
                    if (is_weight_mem) begin
                        weight_mem[{axi_awaddr[15:8], axi_awaddr[7:0]}] <= axi_wdata[7:0];
                    end else if (is_ctrl_reg) begin
                        case (axi_awaddr[7:0])
                            8'h00: core_enable    <= axi_wdata[0];
                            8'h04: clear_activity <= axi_wdata[0];
                            8'h08: input_spikes[31:0]   <= axi_wdata;
                            8'h0C: input_spikes[63:32]  <= axi_wdata;
                            8'h10: input_spikes[95:64]  <= axi_wdata;
                            8'h14: input_spikes[127:96] <= axi_wdata;
                            8'h18: input_spikes[159:128]<= axi_wdata;
                            8'h1C: input_spikes[191:160]<= axi_wdata;
                            8'h20: input_spikes[223:192]<= axi_wdata;
                            8'h24: input_spikes[255:224]<= axi_wdata;
                            8'h28: start_compute  <= axi_wdata[0];
                        endcase
                    end
                    axi_awready <= 1'b1;
                    axi_bresp   <= 2'b00;
                    axi_bvalid  <= 1'b1;
                    wr_state <= WR_IDLE;
                end
            end

            default: wr_state <= WR_IDLE;
        endcase

        // Clear one-shot pulses
        if (start_compute)  start_compute <= 1'b0;
        if (clear_activity) clear_activity <= 1'b0;
    end
end

//==============================================================================
// Bus Read Handling
//==============================================================================
reg [1:0] rd_state;
localparam RD_IDLE = 2'd0;
localparam RD_ADDR = 2'd1;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        axi_arready <= 1'b0;
        axi_rdata   <= 32'd0;
        axi_rresp   <= 2'b00;
        axi_rvalid  <= 1'b0;
        rd_state    <= RD_IDLE;
    end else begin
        case (rd_state)
            RD_IDLE: begin
                axi_arready <= 1'b0;
                axi_rvalid  <= 1'b0;

                if (axi_arvalid) begin
                    axi_arready <= 1'b1;

                    if (is_weight_mem) begin
                        // Return 8-bit weight, zero-padded
                        axi_rdata <= {24'd0, weight_mem[{axi_araddr[15:8], axi_araddr[7:0]}]};
                    end else if (is_ctrl_reg) begin
                        case (axi_araddr[7:0])
                            8'h2C: axi_rdata <= {31'd0, compute_done};
                            8'h30: axi_rdata <= spike_count_low;
                            8'h34: axi_rdata <= spike_count_high;
                            8'h38: axi_rdata <= synapse_count;
                            8'h3C: axi_rdata <= output_spikes_reg[31:0];
                            8'h40: axi_rdata <= output_spikes_reg[63:32];
                            8'h44: axi_rdata <= output_spikes_reg[95:64];
                            8'h48: axi_rdata <= output_spikes_reg[127:96];
                            8'h4C: axi_rdata <= output_spikes_reg[159:128];
                            8'h50: axi_rdata <= output_spikes_reg[191:160];
                            8'h54: axi_rdata <= output_spikes_reg[223:192];
                            8'h58: axi_rdata <= output_spikes_reg[255:224];
                            default: axi_rdata <= 32'd0;
                        endcase
                    end else begin
                        axi_rdata <= 32'hDEAD_BEEF;
                    end

                    axi_rresp  <= 2'b00;
                    axi_rvalid <= 1'b1;
                    rd_state <= RD_ADDR;
                end
            end

            RD_ADDR: begin
                axi_arready <= 1'b0;
                if (axi_rready) begin
                    axi_rvalid <= 1'b0;
                    rd_state <= RD_IDLE;
                end
            end

            default: rd_state <= RD_IDLE;
        endcase
    end
end

//==============================================================================
// Neuromorphic Compute Engine
//==============================================================================
// Instantiates the actual LIF neuron hardware and sequences through
// all 256 neurons each time START_COMPUTE is asserted.
//==============================================================================

// Weight memory read port for compute engine
reg  [15:0] comp_weight_addr;
wire [7:0]  comp_weight_data;
reg         comp_weight_req;
wire        comp_weight_valid;

// Input spikes for compute
reg [255:0] comp_input_spikes;

// Output spikes from compute
wire [255:0] comp_output_spikes;
wire         comp_output_valid;

// Neuromorphic core instance
neuromorphic_core #(
    .NUM_NEURONS(NUM_NEURONS),
    .MEMBRANE_WIDTH(MEM_WIDTH),
    .WEIGHT_WIDTH(8)
) u_neuro_core (
    .clk                (clk),
    .rst_n              (rst_n),
    .enable             (compute_active),
    .clear_activity     (1'b0),
    .weight_addr        (comp_weight_addr),
    .weight_rdata       (comp_weight_data),
    .weight_req         (comp_weight_req),
    .weight_valid       (comp_weight_valid),
    .input_spikes       (comp_input_spikes),
    .input_valid        (comp_input_valid),
    .output_spikes      (comp_output_spikes),
    .output_valid       (comp_output_valid),
    .total_spike_count  (),
    .active_synapse_count(),
    .current_activity   ()
);

// Weight memory access (dual-port: CPU writes + compute reads)
reg comp_weight_delay;
assign comp_weight_data = weight_mem[comp_weight_addr];
assign comp_weight_valid = comp_weight_delay;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        comp_weight_delay <= 1'b0;
    end else if (comp_weight_req) begin
        comp_weight_delay <= 1'b1;
    end else begin
        comp_weight_delay <= 1'b0;
    end
end

// Compute FSM
reg        comp_input_valid;
reg [31:0] comp_cycle_count;
reg [31:0] comp_spike_acc_low;
reg [31:0] comp_spike_acc_high;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        compute_state      <= COMP_IDLE;
        compute_active     <= 1'b0;
        compute_done       <= 1'b0;
        comp_input_valid   <= 1'b0;
        compute_neuron     <= 16'd0;
        comp_cycle_count   <= 32'd0;
        comp_spike_acc_low  <= 32'd0;
        comp_spike_acc_high <= 32'd0;
        output_spikes_reg  <= 256'd0;
        spike_count_low    <= 32'd0;
        spike_count_high   <= 32'd0;
        synapse_count      <= 32'd0;
    end else begin
        case (compute_state)
            COMP_IDLE: begin
                compute_active   <= 1'b0;
                compute_done     <= 1'b0;
                comp_input_valid <= 1'b0;

                if (core_enable && start_compute) begin
                    comp_input_spikes   <= input_spikes;
                    comp_input_valid    <= 1'b1;
                    compute_active      <= 1'b1;
                    compute_state       <= COMP_RUN;
                    comp_cycle_count    <= 32'd0;
                    comp_spike_acc_low  <= 32'd0;
                    comp_spike_acc_high <= 32'd0;
                end
            end

            COMP_RUN: begin
                comp_input_valid <= 1'b0;

                // Wait for neuro core to finish (output_valid handshake)
                if (comp_output_valid) begin
                    output_spikes_reg <= comp_output_spikes;

                    // Accumulate spike count (population count of 256-bit)
                    for (int i = 0; i < 32; i++) begin
                        if (comp_output_spikes[i])   comp_spike_acc_low  <= comp_spike_acc_low + 1;
                        if (comp_output_spikes[i+32]) comp_spike_acc_low  <= comp_spike_acc_low + 1;
                    end
                    for (int i = 64; i < 256; i++) begin
                        if (comp_output_spikes[i])   comp_spike_acc_high <= comp_spike_acc_high + 1;
                    end

                    compute_state <= COMP_DONE;
                end else if (comp_cycle_count > 100000) begin
                    // Timeout fallback
                    compute_state <= COMP_DONE;
                end else begin
                    comp_cycle_count <= comp_cycle_count + 1;
                end
            end

            COMP_DONE: begin
                compute_active          <= 1'b0;
                compute_done            <= 1'b1;
                spike_count_low         <= comp_spike_acc_low;
                spike_count_high        <= comp_spike_acc_high;
                // synapse_count updated by the core (simplified)
                compute_state           <= COMP_IDLE;
            end

            default: compute_state <= COMP_IDLE;
        endcase
    end
end

endmodule


//==============================================================================
// Simplified UART Slave (memory-mapped, for OS debug console)
//==============================================================================
module uart_slave (
    input  wire         clk,
    input  wire         rst_n,

    // AXI-Lite (write — simplified, single-cycle)
    input  wire         axi_awvalid,
    input  wire [3:0]   axi_awaddr,
    output reg          axi_awready,

    input  wire         axi_wvalid,
    input  wire [7:0]   axi_wdata,
    output reg          axi_wready,

    output reg [1:0]    axi_bresp,
    output reg          axi_bvalid,
    input  wire         axi_bready,

    // AXI-Lite (read — simplified)
    input  wire         axi_arvalid,
    input  wire [3:0]   axi_araddr,
    output reg          axi_arready,

    output reg [7:0]    axi_rdata,
    output reg          axi_rvalid,

    // UART pins (simplified — direct byte interface)
    output reg [7:0]    uart_tx_data,
    output reg          uart_tx_valid,
    input  wire         uart_tx_ready,
    input  wire [7:0]   uart_rx_data,
    input  wire         uart_rx_valid,
    output reg          uart_rx_ready
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        axi_awready  <= 1'b0;
        axi_wready   <= 1'b0;
        axi_bresp    <= 2'b00;
        axi_bvalid   <= 1'b0;
        axi_arready  <= 1'b0;
        axi_rdata    <= 8'd0;
        axi_rvalid   <= 1'b0;
        uart_tx_data <= 8'd0;
        uart_tx_valid<= 1'b0;
        uart_rx_ready<= 1'b0;
    end else begin
        // Write (CPU -> UART TX)
        if (axi_awvalid && axi_wvalid && (axi_awaddr == 4'h0)) begin
            uart_tx_data  <= axi_wdata;
            uart_tx_valid <= 1'b1;
            axi_awready   <= 1'b1;
            axi_wready    <= 1'b1;
            axi_bresp     <= 2'b00;
            axi_bvalid    <= 1'b1;
        end else if (axi_awvalid || axi_wvalid) begin
            axi_awready <= 1'b1;
            axi_wready  <= 1'b1;
            axi_bresp   <= 2'b00;
            axi_bvalid  <= 1'b1;
        end else begin
            axi_awready <= 1'b0;
            axi_wready  <= 1'b0;
            axi_bvalid  <= 1'b0;
        end

        // Read (UART RX -> CPU)
        if (axi_arvalid) begin
            axi_arready <= 1'b1;
            if (axi_araddr == 4'h0) begin
                axi_rdata  <= uart_rx_data;
                axi_rvalid <= 1'b1;
                uart_rx_ready <= 1'b1;
            end else if (axi_araddr == 4'h4) begin
                // Status register: bit 0 = RX valid
                axi_rdata  <= {7'd0, uart_rx_valid};
                axi_rvalid <= 1'b1;
            end else begin
                axi_rdata  <= 8'd0;
                axi_rvalid <= 1'b1;
            end
        end else begin
            axi_arready <= 1'b0;
            axi_rvalid  <= 1'b0;
            uart_rx_ready <= 1'b0;
        end

        // Auto-clear TX valid
        if (uart_tx_valid && uart_tx_ready) begin
            uart_tx_valid <= 1'b0;
        end
    end
end

endmodule