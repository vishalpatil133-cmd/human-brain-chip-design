import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class EnergyParams:
    """Energy consumption parameters for neuromorphic operations (in picojoules)"""
    synaptic_op: float = 0.5      # Energy per synaptic operation (spike * weight)
    leak_op: float = 0.01         # Energy per membrane leak update
    spike_gen: float = 0.1        # Energy per spike generation
    reset_op: float = 0.02        # Energy per membrane reset
    memory_read: float = 0.2      # Energy per weight memory read
    
    # Dense GPU/CPU comparison (MAC operation)
    dense_mac: float = 3.5        # Energy per multiply-accumulate (MAC) in pJ

@dataclass
class LIFParams:
    """Leaky Integrate-and-Fire neuron parameters"""
    tau_mem: float = 20.0         # Membrane time constant (ms)
    v_threshold: float = 1.0      # Spike threshold (normalized)
    v_reset: float = 0.0          # Reset potential
    v_rest: float = 0.0           # Resting potential
    refractory_period: float = 2.0 # Refractory period (ms)
    dt: float = 0.1               # Simulation timestep (ms)
    
    # Hardware quantization (8-bit)
    weight_bits: int = 8
    membrane_bits: int = 8

class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron model matching Verilog RTL behavior.
    Implements: leaky integration, threshold detection, spike emission, refractory period.
    """
    
    def __init__(self, params: LIFParams = None, energy_params: EnergyParams = None):
        self.params = params or LIFParams()
        self.energy = energy_params or EnergyParams()
        
        # Neuron state
        self.v_membrane = self.params.v_rest
        self.refractory_counter = 0
        self.spike_count = 0
        self.synaptic_ops = 0
        self.leak_ops = 0
        self.reset_ops = 0
        
        # Precompute leak factor
        self.alpha = np.exp(-self.params.dt / self.params.tau_mem)
        self.beta = 1.0 - self.alpha  # Input integration factor
        
    def reset_state(self):
        """Reset neuron to initial state"""
        self.v_membrane = self.params.v_rest
        self.refractory_counter = 0
        self.spike_count = 0
        self.synaptic_ops = 0
        self.leak_ops = 0
        self.reset_ops = 0
        
    def step(self, input_current: float) -> bool:
        """
        Single simulation step.
        Returns True if neuron spiked.
        """
        spiked = False
        
        # Refractory period handling
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.v_membrane = self.params.v_reset
            return False
            
        # Leaky integration: V = V * alpha + I * beta
        self.v_membrane = self.v_membrane * self.alpha + input_current * self.beta
        self.leak_ops += 1
        
        # Threshold detection
        if self.v_membrane >= self.params.v_threshold:
            spiked = True
            self.spike_count += 1
            self.v_membrane = self.params.v_reset
            self.refractory_counter = int(self.params.refractory_period / self.params.dt)
            self.reset_ops += 1
            
        return spiked
    
    def process_spikes(self, spike_inputs: np.ndarray, weights: np.ndarray) -> bool:
        """
        Process incoming spikes through weighted synapses.
        spike_inputs: binary array of incoming spikes (1 or 0)
        weights: synaptic weights for each input
        """
        # Count active synapses (sparse computation)
        active_synapses = np.where(spike_inputs > 0)[0]
        self.synaptic_ops += len(active_synapses)
        
        if len(active_synapses) == 0:
            input_current = 0.0
        else:
            # Weighted sum of active inputs only (event-driven)
            input_current = np.sum(weights[active_synapses] * spike_inputs[active_synapses])
            
        return self.step(input_current)
    
    def get_energy_consumption(self) -> float:
        """Calculate total dynamic energy consumption in picojoules"""
        energy = (self.synaptic_ops * self.energy.synaptic_op +
                  self.leak_ops * self.energy.leak_op +
                  self.spike_count * self.energy.spike_gen +
                  self.reset_ops * self.energy.reset_op)
        return energy
    
    def get_stats(self) -> dict:
        """Return neuron statistics"""
        return {
            'spike_count': self.spike_count,
            'synaptic_ops': self.synaptic_ops,
            'leak_ops': self.leak_ops,
            'reset_ops': self.reset_ops,
            'energy_pj': self.get_energy_consumption(),
            'final_v_membrane': self.v_membrane
        }

class LIFLayer:
    """Layer of LIF neurons with shared parameters"""
    
    def __init__(self, num_neurons: int, params: LIFParams = None, energy_params: EnergyParams = None):
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron(params, energy_params) for _ in range(num_neurons)]
        self.params = params or LIFParams()
        self.energy = energy_params or EnergyParams()
        
    def reset_all(self):
        for n in self.neurons:
            n.reset_state()
            
    def step_layer(self, input_currents: np.ndarray) -> np.ndarray:
        """Process one timestep for all neurons"""
        spikes = np.zeros(self.num_neurons, dtype=bool)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.step(input_currents[i] if i < len(input_currents) else 0.0)
        return spikes
    
    def process_spikes_batch(self, spike_inputs: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
        """Process spikes for entire layer (event-driven)"""
        spikes = np.zeros(self.num_neurons, dtype=bool)
        for i, neuron in enumerate(self.neurons):
            if i < weight_matrix.shape[0]:
                spikes[i] = neuron.process_spikes(spike_inputs, weight_matrix[i])
        return spikes
    
    def get_total_energy(self) -> float:
        return sum(n.get_energy_consumption() for n in self.neurons)
    
    def get_total_spikes(self) -> int:
        return sum(n.spike_count for n in self.neurons)
    
    def get_stats(self) -> dict:
        total_spikes = self.get_total_spikes()
        total_energy = self.get_total_energy()
        return {
            'num_neurons': self.num_neurons,
            'total_spikes': total_spikes,
            'total_energy_pj': total_energy,
            'avg_spikes_per_neuron': total_spikes / self.num_neurons if self.num_neurons > 0 else 0,
            'energy_per_spike_pj': total_energy / total_spikes if total_spikes > 0 else 0
        }