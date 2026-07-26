import numpy as np
from typing import Tuple, Optional
from neuron_model import LIFLayer, LIFParams, EnergyParams, LIFNeuron

class CrossbarCore:
    """
    Digital neuromorphic crossbar simulator.
    Models a 256-neuron core with SRAM-style weight memory and event-driven updates.
    Tracks dynamic energy consumption vs traditional dense MAC operations.
    """
    
    def __init__(self, num_inputs: int = 256, num_neurons: int = 256,
                 lif_params: Optional[LIFParams] = None,
                 energy_params: Optional[EnergyParams] = None,
                 weight_bits: int = 8):
        
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weight_bits = weight_bits
        
        # Initialize neuron layer
        self.neuron_layer = LIFLayer(num_neurons, lif_params, energy_params)
        self.params = lif_params or LIFParams()
        self.energy = energy_params or EnergyParams()
        
        # Weight matrix (8-bit quantized)
        self.weights = np.random.randint(-2**(weight_bits-1), 2**(weight_bits-1)-1, 
                                        (num_neurons, num_inputs)).astype(np.int8)
        # Normalize weights to [-1, 1] range
        self.weights_normalized = self.weights / (2**(weight_bits-1))
        
        # Activity tracking
        self.activity_trace = []  # Spike activity per timestep
        self.energy_trace = []    # Energy per timestep
        
    def reset(self):
        """Reset core state"""
        self.neuron_layer.reset_all()
        self.activity_trace = []
        self.energy_trace = []
        
    def step(self, input_spikes: np.ndarray) -> np.ndarray:
        """Process one timestep with given input spikes"""
        # Event-driven computation: only active synapses processed
        output_spikes = self.neuron_layer.process_spikes_batch(input_spikes, self.weights_normalized)
        
        # Track activity
        spike_activity = np.sum(output_spikes)
        self.activity_trace.append(spike_activity)
        
        # Track energy
        total_energy = sum(n.get_energy_consumption() for n in self.neuron_layer.neurons)
        self.energy_trace.append(total_energy)
        
        return output_spikes
    
    def run_timesteps(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Run simulation for multiple timesteps.
        spike_trains shape: (timesteps, num_inputs)
        """
        self.reset()
        timesteps = spike_trains.shape[0]
        output_spikes = np.zeros((timesteps, self.num_neurons), dtype=bool)
        
        for t in range(timesteps):
            output_spikes[t] = self.step(spike_trains[t])
            
        return output_spikes
    
    def get_sparsity(self) -> float:
        """Calculate input spike sparsity"""
        total_potential = self.num_inputs * len(self.activity_trace)
        actual_spikes = sum(self.activity_trace)
        return 1.0 - (actual_spikes / total_potential) if total_potential > 0 else 1.0
    
    def get_neuromorphic_energy(self) -> float:
        """Total energy consumed by neuromorphic core (pJ)"""
        return sum(self.energy_trace)
    
    def get_dense_energy_equivalent(self) -> float:
        """
        Calculate equivalent energy if dense MAC operations were used.
        Dense approach: for each timestep, compute all input*weight products regardless of spike activity.
        """
        timesteps = len(self.activity_trace)
        # Dense: all inputs x all neurons x all timesteps
        total_macs = timesteps * self.num_inputs * self.num_neurons
        return total_macs * self.energy.dense_mac
    
    def get_energy_comparison(self) -> dict:
        """Return detailed energy comparison"""
        neuro_energy = self.get_neuromorphic_energy()
        dense_energy = self.get_dense_energy_equivalent()
        
        # Also calculate energy for parameter-matched GPU/CPU
        gpu_energy = dense_energy
        total_spikes = np.sum(self.activity_trace) if self.activity_trace else 0
        
        return {
            'neuromorphic_energy_pj': neuro_energy,
            'dense_mac_energy_pj': dense_energy,
            'energy_savings_pct': (1 - neuro_energy / dense_energy) * 100 if dense_energy > 0 else 0,
            'energy_efficiency_ratio': dense_energy / neuro_energy if neuro_energy > 0 else float('inf'),
            'total_spikes': total_spikes,
            'sparsity_pct': self.get_sparsity() * 100,
            'timesteps': len(self.activity_trace),
            'num_neurons': self.num_neurons,
            'num_inputs': self.num_inputs
        }


class TemporalPatternClassifier:
    """
    Temporal pattern classification demo using neuromorphic core.
    Detects specific timing patterns in sensory spike inputs.
    """
    
    def __init__(self, num_input_neurons: int = 64, num_output_neurons: int = 4):
        self.core = CrossbarCore(
            num_inputs=num_input_neurons,
            num_neurons=num_output_neurons,
            lif_params=LIFParams(
                tau_mem=10.0,
                v_threshold=0.8,
                v_reset=0.0,
                refractory_period=1.0,
                dt=0.1
            )
        )
        
    def generate_pattern(self, pattern_id: int, timesteps: int = 100) -> np.ndarray:
        """Generate synthetic spike pattern for given class"""
        spike_train = np.zeros((timesteps, 64))
        
        if pattern_id == 0:  # Early burst pattern
            for t in range(10, 30):
                if np.random.random() < 0.3:
                    spike_train[t, 0:16] = 1.0
                    
        elif pattern_id == 1:  # Late burst pattern
            for t in range(60, 80):
                if np.random.random() < 0.3:
                    spike_train[t, 16:32] = 1.0
                    
        elif pattern_id == 2:  # Rhythmic pattern
            for t in range(0, timesteps):
                if t % 10 < 3 and np.random.random() < 0.3:
                    spike_train[t, 32:48] = 1.0
                    
        else:  # Random noise pattern
            for t in range(0, timesteps):
                if np.random.random() < 0.05:
                    spike_train[t, 48:64] = 1.0
                    
        return spike_train
    
    def classify(self, spike_train: np.ndarray) -> int:
        """Run classification and return winning neuron index"""
        output_spikes = self.core.run_timesteps(spike_train)
        spike_counts = np.sum(output_spikes, axis=0)
        return np.argmax(spike_counts), spike_counts
    
    def run_benchmark(self, num_samples: int = 20, timesteps: int = 100) -> dict:
        """Run classification benchmark on all 4 patterns"""
        correct = 0
        total = 4 * num_samples
        
        for pattern_id in range(4):
            for _ in range(num_samples):
                spike_train = self.generate_pattern(pattern_id, timesteps)
                predicted, _ = self.classify(spike_train)
                if predicted == pattern_id:
                    correct += 1
                    
        return {
            'accuracy': correct / total,
            'correct': correct,
            'total': total
        }


def generate_poisson_spike_trains(num_inputs: int, timesteps: int, 
                                   firing_rate: float = 0.1) -> np.ndarray:
    """
    Generate Poisson-distributed spike trains.
    Each input independently generates spikes with given probability per timestep.
    """
    return (np.random.random((timesteps, num_inputs)) < firing_rate).astype(float)


def generate_correlated_spike_trains(num_inputs: int, timesteps: int,
                                      correlation: float = 0.5, 
                                      base_rate: float = 0.1) -> np.ndarray:
    """
    Generate spike trains with controlled correlation structure.
    Useful for testing pattern detection capabilities.
    """
    # Common input drives all neurons
    common = np.random.random(timesteps) < base_rate
    # Independent noise per neuron
    independent = np.random.random((timesteps, num_inputs)) < base_rate
    
    spike_trains = np.zeros((timesteps, num_inputs))
    for i in range(num_inputs):
        for t in range(timesteps):
            if np.random.random() < correlation:
                spike_trains[t, i] = float(common[t])
            else:
                spike_trains[t, i] = float(independent[t, i])
                
    return spike_trains


def run_sparsity_sweep(num_trials: int = 5) -> dict:
    """Run energy comparison across different sparsity levels"""
    firing_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = {}
    
    for rate in firing_rates:
        neuro_energies = []
        dense_energies = []
        
        for _ in range(num_trials):
            core = CrossbarCore(num_inputs=128, num_neurons=128)
            spikes = generate_poisson_spike_trains(128, 200, rate)
            core.run_timesteps(spikes)
            comp = core.get_energy_comparison()
            neuro_energies.append(comp['neuromorphic_energy_pj'])
            dense_energies.append(comp['dense_mac_energy_pj'])
            
        results[f'rate_{rate}'] = {
            'firing_rate': rate,
            'neuromorphic_energy_pj': np.mean(neuro_energies),
            'dense_energy_pj': np.mean(dense_energies),
            'efficiency_ratio': np.mean(dense_energies) / np.mean(neuro_energies) if np.mean(neuro_energies) > 0 else 0,
            'sparsity': 1.0 - rate
        }
        
    return results