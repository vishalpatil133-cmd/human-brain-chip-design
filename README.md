# SNN Multi-Model Brain Chip Simulator & RTL Design

**Architect & Developer**: [Vishal Patil](https://www.linkedin.com/in/vishal-patil-928600340/) | **GitHub**: [@vishalpatil133-cmd](https://github.com/vishalpatil133-cmd)

Inspired by the human brain, this project models and simulates a digital neuromorphic processor core. Modern deep learning hardware (like GPUs and TPUs) is bottlenecked by the von Neumann memory wall and consumes high levels of energy by executing dense, continuous **Multiply-Accumulate (MAC)** operations. 

In contrast, our neuromorphic chip achieves massive energy efficiency by mimicking biological brains using:
1. **Spiking Neural Networks (SNNs)**: Information is transmitted via binary events (spikes) rather than continuous high-precision activation values.
2. **Leaky Integrate-and-Fire (LIF) Neurons**: Digital neurons integrate input spikes over time, leak energy slowly, and fire an output spike only when crossing a threshold.
3. **Event-Driven Sparsity**: Neurons and synapses only update when spikes are active, reducing dynamic power to near-zero during idle periods.

This SNN simulation is scaled to **25,000 Inputs and 25,000 Neurons** using **1% Synaptic Sparsity** (6.25 million active synapses), demonstrating a colossal hardware advantage of **62.5 Billion operations avoided** and **99.9996% dynamic energy savings**.

---

## Triple-Chiplet Multi-Chip Module (MCM) Architecture

To demonstrate an industry-standard modern ASIC layout, we have designed a **Triple-Chiplet System-on-Package** connected via a high-bandwidth die-to-die physical interface. This partitions logic optimizes fabrication costs:

```
+------------------------------------------------------------------------------------------------+
|                                   Triple-Chiplet MCM Package                                   |
|                                                                                                |
|   +---------------------------------------+                +-------------------------------+   |
|   |         PERFORMANCE CHIPLET           |                |     NEUROMORPHIC CHIPLET      |   |
|   |                                       |                |                               |   |
|   |   +-------------+   +-------------+   |  UCIe Link Bus |   +-----------------------+   |   |
|   |   |   RISC-V    |   | 5G Modem    |   |  ==== 16-bit   |   | 2D Mesh SNN Tile Grid |   |   |
|   |   |   CPU Core  |   | Accelerator |   |  D2D lanes === |   | (4x4 Router Mesh Array|   |   |
|   |   +-------------+   +-------------+   |       ||       |   |  LIF Neurons + BRAM)  |   |   |
|   |                                       |       ||       |   +-----------------------+   |   |
|   +---------------------------------------+       ||       +-------------------------------+   |
|                                                   ||                                           |
|                                                   ||       +-------------------------------+   |
|                                                   +======> |          GPU CHIPLET          |   |
|                                                            |  (Rasterizer + Color Blender) |   |
|                                                            +-------------------------------+   |
|                                                                                                |
+------------------------------------------------------------------------------------------------+
```

### 1. Performance Chiplet (Die A - Logic Optimized)
- **RISC-V Core**: Single-cycle RV32I processor coordinate boot firmware.
- **5G Modem Accelerator**: Integrated hardware pipeline for legacy 5G Modem FFT operations.
- **UCIe Master Bridge**: Direct Memory-Mapped Bridge translating internal memory accesses targeting SNN or GPU spaces into physical link packets.

### 2. Neuromorphic Chiplet (Die B - High-Density RAM Optimized)
- **Tile Array**: 4x4 grid of SNN tiles (scaleable up to 25,000 neurons) utilizing **Network-on-Chip (NoC)**.
- **Router Logic**: Spikes are routed as Address Event Representation (AER) packets through an X-Y mesh routing network to local LIF neurons.
- **UCIe Slave Bridge**: Receives incoming config write packets from Die A and decodes them into local register banks.

### 3. GPU Chiplet (Die C - Graphics Rendering Core)
- **Rasterizer Engine**: Coordinates 2D graphics rendering and vertex coordinate storage.
- **Pixel Blender**: Fast parallel hardware blender: $C_{\text{out}} = (C_A \cdot \alpha + C_B \cdot (255 - \alpha)) / 255$.
- **UCIe Slave Bridge**: Maps graphics rendering configurations directly to register addresses.

### 4. Die-to-Die Interconnect (UCIe-Style Link)
- Implemented in [chiplet_link.v](file:///d:/human%20brain%20chip%20design/rtl/chiplet_link.v).
- Serializes 32-bit CPU transactions into 16-bit physical interposer words.
- Features parallel TX/RX channels, credit-based flow control, and data packet framing with destination ID routing.

---

## Directory Structure

```
├── rtl/
│   ├── triple_chiplet_top.v     # Synthesizable top-level Multi-Chip Module (MCM)
│   ├── performance_chiplet.v    # Synthesizable CPU Chiplet (RISC-V + 5G)
│   ├── neuromorphic_chiplet.v   # Synthesizable SNN Chiplet (AER NoC + Tiles)
│   ├── gpu_chiplet.v            # Synthesizable GPU Chiplet (Color Blender + Rasterizer)
│   ├── chiplet_link.v           # Synthesizable D2D UCIe-style link controller (with router)
│   ├── neuromorphic_soc.v       # Standalone SoC configuration (legacy)
│   └── neuromorphic_core.v      # Synthesizable standalone 16-neuron LIF core
├── src/
│   ├── main_os.c                # Bare-metal C OS running on RISC-V CPU
│   ├── NeuronModel.java         # Cycle-accurate Java model of the LIF neuron
│   ├── CrossbarSimulator.java   # Sparse synaptic list simulation (25K inputs/neurons)
│   └── RunSimulation.java       # SNN simulator driving 25K sparse neuron simulation
├── simulation_results.json      # Output log containing all step states and energy metrics
├── simulation_results.js        # JavaScript wrapper of results (used by visualizer to avoid CORS)
├── visualizer.html              # Sleek HTML5 dashboard with 25K density canvases
└── README.md                    # Project documentation (this file)
```

---

## Getting Started

### 1. Run the 25K SNN Simulator
```bash
javac src/NeuronModel.java src/CrossbarSimulator.java src/RunSimulation.java
java src.RunSimulation
```

### 2. View the Interactive Dashboard
Open `visualizer.html` directly in any web browser.

### 3. Compile/Synthesize Verilog Chiplets
Compile the top-level MCM Verilog files:
```bash
# Verify syntax of all chiplets using a simulator or synthesis tool
# Files needed:
# - rtl/triple_chiplet_top.v
# - rtl/performance_chiplet.v
# - rtl/neuromorphic_chiplet.v
# - rtl/gpu_chiplet.v
# - rtl/chiplet_link.v
```
