import numpy as np
from typing import Tuple, Optional, List, Dict
from neuron_model import LIFNeuron, LIFLayer, LIFParams, EnergyParams
from crossbar_simulator import CrossbarCore
from encoding import ImageToSpikeEncoder, AudioToSpikeEncoder


class SNNClassifier:
    """
    Spiking Neural Network classifier for image/audio recognition.
    Implements a two-layer SNN: input encoding -> hidden LIF layer -> output LIF layer.
    Trained via surrogate gradient / reward-based learning simulation.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 timesteps: int = 100, learning_rate: float = 0.01):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        
        # Synaptic weight matrices
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        
        # Neuron parameters
        self.hidden_params = LIFParams(
            tau_mem=15.0, v_threshold=1.0, v_reset=0.0,
            refractory_period=1.0, dt=0.1
        )
        self.output_params = LIFParams(
            tau_mem=10.0, v_threshold=0.8, v_reset=0.0,
            refractory_period=0.5, dt=0.1
        )
        
        # Energy tracking
        self.energy_used = 0.0
        self.dense_energy_equiv = 0.0
        
    def forward(self, input_spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Forward pass through the SNN.
        
        Args:
            input_spikes: (timesteps, input_size) binary spike train
            
        Returns:
            hidden_spikes: (timesteps, hidden_size)
            output_spikes: (timesteps, output_size)
            energy_stats: dict with energy comparison
        """
        timesteps = input_spikes.shape[0]
        
        # Initialize neuron layers
        hidden_layer = LIFLayer(self.hidden_size, self.hidden_params)
        output_layer = LIFLayer(self.output_size, self.output_params)
        
        # Storage for spike traces
        hidden_spikes = np.zeros((timesteps, self.hidden_size))
        output_spikes = np.zeros((timesteps, self.output_size))
        
        # Event-driven computation
        for t in range(timesteps):
            # Input to hidden (event-driven)
            input_t = input_spikes[t]
            active_inputs = np.where(input_t > 0)[0]
            
            for h in range(self.hidden_size):
                if len(active_inputs) > 0:
                    input_current = np.sum(self.W1[h, active_inputs] * input_t[active_inputs])
                else:
                    input_current = 0.0
                spiked = hidden_layer.neurons[h].step(input_current)
                if spiked:
                    hidden_spikes[t, h] = 1.0
            
            # Hidden to output (event-driven)
            active_hidden = np.where(hidden_spikes[t] > 0)[0]
            
            for o in range(self.output_size):
                if len(active_hidden) > 0:
                    input_current = np.sum(self.W2[o, active_hidden] * hidden_spikes[t, active_hidden])
                else:
                    input_current = 0.0
                spiked = output_layer.neurons[o].step(input_current)
                if spiked:
                    output_spikes[t, o] = 1.0
        
        # Calculate energy comparison
        neuro_energy = (hidden_layer.get_total_energy() + output_layer.get_total_energy())
        n_hidden_ops = timesteps * self.hidden_size * self.input_size
        n_output_ops = timesteps * self.output_size * self.hidden_size
        dense_energy = (n_hidden_ops + n_output_ops) * 3.5  # 3.5 pJ per MAC
        
        self.energy_used = neuro_energy
        self.dense_energy_equiv = dense_energy
        
        energy_stats = {
            'neuromorphic_energy_pj': neuro_energy,
            'dense_mac_energy_pj': dense_energy,
            'efficiency_ratio': dense_energy / neuro_energy if neuro_energy > 0 else float('inf'),
            'savings_pct': (1 - neuro_energy / dense_energy) * 100 if dense_energy > 0 else 0
        }
        
        return hidden_spikes, output_spikes, energy_stats
    
    def predict(self, input_spikes: np.ndarray) -> int:
        """Classify input and return predicted class label"""
        _, output_spikes, _ = self.forward(input_spikes)
        spike_counts = np.sum(output_spikes, axis=0)
        return np.argmax(spike_counts)
    
    def train_sample(self, input_spikes: np.ndarray, target: int):
        """
        Train on a single sample using reward-modulated plasticity.
        Simplified STDP-like learning rule.
        """
        hidden_spikes, output_spikes, _ = self.forward(input_spikes)
        
        output_counts = np.sum(output_spikes, axis=0)
        predicted = np.argmax(output_counts)
        
        # Reward signal: +1 if correct, -1 if incorrect
        reward = 1.0 if predicted == target else -0.5
        
        # Update weights based on spike-timing correlations
        # Pre-synaptic trace (exponential decay)
        pre_trace = np.zeros(self.hidden_size)
        decay = 0.9
        
        for t in range(self.timesteps):
            # Update pre-synaptic trace
            pre_trace = decay * pre_trace + hidden_spikes[t]
            
            # For each output neuron
            for o in range(self.output_size):
                if output_spikes[t, o] > 0:
                    post_target = 1.0 if o == target else -0.5
                    # STDP: weight update based on pre-trace
                    delta = reward * post_target * self.learning_rate * pre_trace
                    self.W2[o] += delta
                    
                    # Also update hidden weights (simplified)
                    for h in range(self.hidden_size):
                        if hidden_spikes[t, h] > 0:
                            self.W1[h, :self.input_size] += reward * post_target * self.learning_rate * 0.1
        
        # Clip weights
        np.clip(self.W1, -1.0, 1.0, out=self.W1)
        np.clip(self.W2, -1.0, 1.0, out=self.W2)
        
        return predicted == target
    
    def train(self, spike_trains: List[np.ndarray], labels: np.ndarray,
              epochs: int = 10, verbose: bool = True) -> List[float]:
        """Train on dataset"""
        accuracies = []
        n_samples = len(spike_trains)
        
        for epoch in range(epochs):
            correct = 0
            
            # Shuffle
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                result = self.train_sample(spike_trains[idx], labels[idx])
                if result:
                    correct += 1
            
            accuracy = correct / n_samples
            accuracies.append(accuracy)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.3f}")
        
        return accuracies


class ImageRecognitionModel:
    """
    SNN model for image recognition.
    Handles: image -> spike encoding -> SNN classification -> energy analysis.
    """
    
    def __init__(self, img_size: int = 28, hidden_size: int = 128, 
                 num_classes: int = 10, timesteps: int = 100):
        
        self.img_size = img_size
        self.input_size = img_size * img_size
        self.num_classes = num_classes
        self.timesteps = timesteps
        
        self.snn = SNNClassifier(
            input_size=self.input_size,
            hidden_size=hidden_size,
            output_size=num_classes,
            timesteps=timesteps
        )
        
    def classify_image(self, image: np.ndarray, encoding: str = 'poisson') -> Tuple[int, np.ndarray, Dict]:
        """
        Classify a single image.
        
        Returns:
            prediction: class label
            output_spikes: per-class spike activity
            energy_stats: energy comparison dict
        """
        # Encode image to spikes
        spike_train = ImageToSpikeEncoder.encode_image(
            image, self.timesteps, method=encoding
        )
        
        # Run through SNN
        _, output_spikes, energy_stats = self.snn.forward(spike_train)
        
        # Decode output
        spike_counts = np.sum(output_spikes, axis=0)
        prediction = np.argmax(spike_counts)
        
        return prediction, spike_counts, energy_stats
    
    def train_on_dataset(self, images: np.ndarray, labels: np.ndarray,
                         epochs: int = 5, encoding: str = 'poisson') -> List[float]:
        """Train on image dataset"""
        # Pre-encode all images to spike trains
        spike_trains = []
        for img in images:
            spikes = ImageToSpikeEncoder.encode_image(img, self.timesteps, method=encoding)
            spike_trains.append(spikes)
        
        return self.snn.train(spike_trains, labels, epochs)
    
    def evaluate(self, images: np.ndarray, labels: np.ndarray,
                 encoding: str = 'poisson') -> Dict:
        """Evaluate model on test dataset"""
        correct = 0
        total = len(images)
        total_neuro_energy = 0.0
        total_dense_energy = 0.0
        
        for i, img in enumerate(images):
            pred, _, energy = self.classify_image(img, encoding)
            if pred == labels[i]:
                correct += 1
            total_neuro_energy += energy['neuromorphic_energy_pj']
            total_dense_energy += energy['dense_mac_energy_pj']
        
        return {
            'accuracy': correct / total,
            'correct': correct,
            'total': total,
            'avg_neuro_energy_pj': total_neuro_energy / total,
            'avg_dense_energy_pj': total_dense_energy / total,
            'energy_efficiency_ratio': total_dense_energy / total_neuro_energy if total_neuro_energy > 0 else 0,
            'energy_savings_pct': (1 - total_neuro_energy / total_dense_energy) * 100 if total_dense_energy > 0 else 0
        }


class AudioRecognitionModel:
    """
    SNN model for audio/speech command recognition.
    Handles: audio -> feature extraction -> spike encoding -> SNN classification.
    """
    
    def __init__(self, num_features: int = 13, hidden_size: int = 64,
                 num_commands: int = 5, timesteps: int = 100):
        
        self.num_features = num_features
        self.num_commands = num_commands
        self.timesteps = timesteps
        
        self.snn = SNNClassifier(
            input_size=num_features,
            hidden_size=hidden_size,
            output_size=num_commands,
            timesteps=timesteps
        )
        
    def classify_audio(self, audio: np.ndarray, sample_rate: int = 16000,
                       encoding: str = 'poisson') -> Tuple[int, np.ndarray, Dict]:
        """Classify an audio signal"""
        # Encode audio to spikes
        spike_train = AudioToSpikeEncoder.encode_audio(
            audio, sample_rate, self.timesteps, method=encoding,
            num_channels=self.num_features
        )
        
        # Run through SNN
        _, output_spikes, energy_stats = self.snn.forward(spike_train)
        
        # Decode output
        spike_counts = np.sum(output_spikes, axis=0)
        prediction = np.argmax(spike_counts)
        
        return prediction, spike_counts, energy_stats
    
    def train_on_dataset(self, audio_samples: List[np.ndarray], labels: np.ndarray,
                         epochs: int = 5, encoding: str = 'poisson') -> List[float]:
        """Train on audio dataset"""
        spike_trains = []
        for audio in audio_samples:
            spikes = AudioToSpikeEncoder.encode_audio(
                audio, 16000, self.timesteps, method=encoding,
                num_channels=self.num_features
            )
            spike_trains.append(spikes)
        
        return self.snn.train(spike_trains, labels, epochs)
    
    def evaluate(self, audio_samples: List[np.ndarray], labels: np.ndarray,
                 encoding: str = 'poisson') -> Dict:
        """Evaluate on test dataset"""
        correct = 0
        total = len(audio_samples)
        total_neuro_energy = 0.0
        total_dense_energy = 0.0
        
        for i, audio in enumerate(audio_samples):
            pred, _, energy = self.classify_audio(audio, encoding=encoding)
            if pred == labels[i]:
                correct += 1
            total_neuro_energy += energy['neuromorphic_energy_pj']
            total_dense_energy += energy['dense_mac_energy_pj']
        
        return {
            'accuracy': correct / total,
            'correct': correct,
            'total': total,
            'avg_neuro_energy_pj': total_neuro_energy / total,
            'avg_dense_energy_pj': total_dense_energy / total,
            'energy_efficiency_ratio': total_dense_energy / total_neuro_energy if total_neuro_energy > 0 else 0,
            'energy_savings_pct': (1 - total_neuro_energy / total_dense_energy) * 100 if total_dense_energy > 0 else 0
        }


class CrossModalIntegration:
    """
    Demonstrates cross-modal (image + audio) integration on the neuromorphic chip.
    Combines visual and auditory streams into a unified decision.
    """
    
    def __init__(self, img_size: int = 28, audio_features: int = 13,
                 hidden_size: int = 128, num_classes: int = 5, timesteps: int = 100):
        
        self.img_encoder = ImageToSpikeEncoder()
        self.audio_encoder = AudioToSpikeEncoder()
        
        # Multi-modal input: image pixels + audio features
        total_input = img_size * img_size + audio_features
        
        self.snn = SNNClassifier(
            input_size=total_input,
            hidden_size=hidden_size,
            output_size=num_classes,
            timesteps=timesteps
        )
        
    def classify_multimodal(self, image: np.ndarray, audio: np.ndarray,
                            sample_rate: int = 16000) -> Tuple[int, Dict]:
        """Classify using both visual and auditory input"""
        # Encode both modalities
        img_spikes = ImageToSpikeEncoder.encode_image(image, self.snn.timesteps)
        audio_spikes = AudioToSpikeEncoder.encode_audio(
            audio, sample_rate, self.snn.timesteps,
            num_channels=self.audio_encoder._compute_mfcc_like(audio, sample_rate).shape[0]
        )
        
        # Concatenate spike trains (multi-modal fusion)
        combined_spikes = np.concatenate([
            img_spikes,
            np.resize(audio_spikes, (self.snn.timesteps, audio_spikes.shape[1])) 
            if audio_spikes.shape[1] <= img_spikes.shape[1] else
            audio_spikes[:, :img_spikes.shape[1]]
        ], axis=1)
        
        # Classify
        _, output_spikes, energy_stats = self.snn.forward(combined_spikes)
        spike_counts = np.sum(output_spikes, axis=0)
        prediction = np.argmax(spike_counts)
        
        return prediction, energy_stats