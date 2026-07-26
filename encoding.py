import numpy as np
from typing import Tuple, Optional, List

class ImageToSpikeEncoder:
    """
    Converts images to spike trains for neuromorphic processing.
    Supports multiple encoding schemes: Poisson rate coding, temporal coding, rank-order coding.
    """
    
    @staticmethod
    def poisson_rate_coding(image: np.ndarray, timesteps: int, 
                            norm_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """
        Poisson rate coding: pixel intensity -> spike probability per timestep.
        Higher intensity -> more spikes (higher firing rate).
        
        Args:
            image: 2D image array (H, W) or flattened 1D (N,)
            timesteps: Number of simulation timesteps
            norm_range: Normalized value range (min, max)
            
        Returns:
            spike_train: (timesteps, N) binary array
        """
        flat = image.flatten().astype(float)
        flat = np.clip(flat, norm_range[0], norm_range[1])
        
        # Normalize to [0, 1]
        if norm_range[1] > norm_range[0]:
            flat = (flat - norm_range[0]) / (norm_range[1] - norm_range[0])
        
        num_pixels = len(flat)
        spike_train = np.zeros((timesteps, num_pixels), dtype=float)
        
        for t in range(timesteps):
            spike_train[t] = (np.random.random(num_pixels) < flat).astype(float)
            
        return spike_train
    
    @staticmethod
    def temporal_coding(image: np.ndarray, timesteps: int,
                        norm_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """
        Temporal coding: pixel intensity -> spike latency.
        Brighter pixels spike earlier, darker pixels spike later (or not at all).
        Each pixel spikes at most once.
        """
        flat = image.flatten().astype(float)
        flat = np.clip(flat, norm_range[0], norm_range[1])
        
        if norm_range[1] > norm_range[0]:
            flat = (flat - norm_range[0]) / (norm_range[1] - norm_range[0])
        
        num_pixels = len(flat)
        spike_train = np.zeros((timesteps, num_pixels), dtype=float)
        
        # Convert intensity to latency (inverse: high intensity -> early spike)
        # Spike time = (1 - intensity) * (timesteps - 1)
        spike_times = ((1.0 - flat) * (timesteps - 1)).astype(int)
        spike_times = np.clip(spike_times, 0, timesteps - 1)
        
        for i, t in enumerate(spike_times):
            spike_train[t, i] = 1.0
            
        return spike_train
    
    @staticmethod
    def rank_order_coding(image: np.ndarray, timesteps: int,
                          norm_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """
        Rank-order coding: pixels are sorted by intensity and assigned spike times.
        Only top-K pixels spike, or spiking based on rank threshold.
        """
        flat = image.flatten().astype(float)
        flat = np.clip(flat, norm_range[0], norm_range[1])
        
        num_pixels = len(flat)
        spike_train = np.zeros((timesteps, num_pixels), dtype=float)
        
        # Sort indices by intensity (descending)
        sorted_indices = np.argsort(flat)[::-1]
        
        # Assign spike times based on rank
        # Only spike if intensity > median
        threshold = np.median(flat) if norm_range[1] > norm_range[0] else 0.5
        active_count = np.sum(flat > threshold)
        
        for rank, idx in enumerate(sorted_indices[:active_count]):
            # Early ranks spike early
            spike_time = int((rank / max(active_count, 1)) * (timesteps - 1))
            spike_time = min(spike_time, timesteps - 1)
            spike_train[spike_time, idx] = 1.0
            
        return spike_train
    
    @staticmethod
    def encode_image(image: np.ndarray, timesteps: int, 
                     method: str = 'poisson') -> np.ndarray:
        """Convenience method to encode image with specified method"""
        methods = {
            'poisson': ImageToSpikeEncoder.poisson_rate_coding,
            'temporal': ImageToSpikeEncoder.temporal_coding,
            'rank_order': ImageToSpikeEncoder.rank_order_coding
        }
        encoder = methods.get(method, methods['poisson'])
        return encoder(image, timesteps)


class AudioToSpikeEncoder:
    """
    Converts audio signals to spike trains for neuromorphic processing.
    Implements: spectrogram extraction, MFCC-like features, and spike encoding.
    """
    
    @staticmethod
    def generate_synthetic_audio(num_samples: int = 8000, sample_rate: int = 16000) -> np.ndarray:
        """
        Generate synthetic audio signal with frequency components.
        Returns: audio_signal (num_samples,)
        """
        t = np.arange(num_samples) / sample_rate
        
        # Base frequency
        f0 = 440.0  # A4 note
        audio = 0.5 * np.sin(2 * np.pi * f0 * t)
        
        # Add harmonics
        audio += 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        audio += 0.125 * np.sin(2 * np.pi * 3 * f0 * t)
        
        # Add temporal envelope
        envelope = np.exp(-t / 0.5)  # Decay over 0.5s
        audio = audio * envelope
        
        # Add noise
        audio += 0.05 * np.random.randn(num_samples)
        
        return audio
    
    @staticmethod
    def _compute_spectrogram(audio: np.ndarray, sample_rate: int = 16000,
                             win_size: int = 256, hop_length: int = 128) -> np.ndarray:
        """
        Simple spectrogram computation using STFT.
        Returns: (freq_bins, time_frames)
        """
        # Pad audio
        if len(audio) < win_size:
            audio = np.pad(audio, (0, win_size - len(audio)))
            
        num_frames = 1 + (len(audio) - win_size) // hop_length
        freq_bins = win_size // 2 + 1
        
        # Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win_size) / win_size))
        
        spectrogram = np.zeros((freq_bins, num_frames))
        
        for i in range(num_frames):
            start = i * hop_length
            segment = audio[start:start + win_size] * window
            spectrum = np.abs(np.fft.rfft(segment))
            spectrogram[:, i] = spectrum
            
        # Convert to log scale (dB-like)
        spectrogram = np.log1p(spectrogram)
        
        return spectrogram
    
    @staticmethod
    def _compute_mfcc_like(audio: np.ndarray, sample_rate: int = 16000,
                           num_coeffs: int = 13, num_filters: int = 26) -> np.ndarray:
        """Compute MFCC-like features from audio"""
        spectrogram = AudioToSpikeEncoder._compute_spectrogram(audio, sample_rate)
        n_freq, n_time = spectrogram.shape
        
        # Simple mel filterbank approximation
        mel_filters = np.zeros((num_filters, n_freq))
        for m in range(num_filters):
            f_min = int(m * n_freq / num_filters)
            f_max = int((m + 1) * n_freq / num_filters)
            f_center = (f_min + f_max) // 2
            
            for f in range(n_freq):
                if f_min <= f < f_center:
                    mel_filters[m, f] = (f - f_min) / (f_center - f_min)
                elif f_center <= f < f_max:
                    mel_filters[m, f] = (f_max - f) / (f_max - f_center)
        
        # Apply mel filters
        mel_spectrum = mel_filters @ spectrogram
        
        # Apply log and DCT-like transform for MFCCs
        mel_spectrum = np.log1p(np.maximum(mel_spectrum, 1e-10))
        
        # Simple DCT
        mfcc = np.zeros((num_coeffs, n_time))
        for k in range(num_coeffs):
            mfcc[k] = np.sum(mel_spectrum * np.cos(np.pi * k * (np.arange(num_filters) + 0.5) / num_filters), axis=0)
            
        return mfcc
    
    @staticmethod
    def encode_audio(audio: Optional[np.ndarray] = None, sample_rate: int = 16000,
                     timesteps: int = 100, method: str = 'poisson',
                     num_channels: int = 13) -> np.ndarray:
        """
        Convert audio to spike trains.
        
        Args:
            audio: Audio signal, or None for synthetic audio
            sample_rate: Audio sample rate in Hz
            timesteps: Number of timesteps for spike encoding
            method: 'poisson', 'temporal', or 'rank_order'
            num_channels: Number of frequency/MFCC channels
            
        Returns:
            spike_train: (timesteps, num_channels) binary array
        """
        if audio is None:
            audio = AudioToSpikeEncoder.generate_synthetic_audio(sample_rate * 2)
        
        # Extract features
        mfcc = AudioToSpikeEncoder._compute_mfcc_like(audio, sample_rate, num_coeffs=num_channels)
        
        # Repeat or interpolate MFCC to match timesteps
        n_frames = mfcc.shape[1]
        feature_matrix = np.zeros((num_channels, timesteps))
        
        for i in range(num_channels):
            feature_matrix[i] = np.interp(
                np.linspace(0, n_frames - 1, timesteps),
                np.arange(n_frames),
                mfcc[i]
            )
        
        # Normalize features to [0, 1]
        f_min = feature_matrix.min()
        f_max = feature_matrix.max()
        if f_max > f_min:
            feature_matrix = (feature_matrix - f_min) / (f_max - f_min)
        else:
            feature_matrix = np.zeros_like(feature_matrix)
        
        # Encode features as spike trains
        # Transpose to (timesteps, num_channels)
        spike_train = np.zeros((timesteps, num_channels), dtype=float)
        
        if method == 'poisson':
            for t in range(timesteps):
                rates = np.clip(feature_matrix[:, t], 0, 1)
                spike_train[t] = (np.random.random(num_channels) < rates).astype(float)
        elif method == 'temporal':
            for ch in range(num_channels):
                spike_time = int((1.0 - feature_matrix[ch].mean()) * (timesteps - 1))
                spike_time = np.clip(spike_time, 0, timesteps - 1)
                spike_train[spike_time, ch] = 1.0
        elif method == 'rank_order':
            avg_features = feature_matrix.mean(axis=1)
            sorted_channels = np.argsort(avg_features)[::-1]
            active = np.sum(avg_features > np.median(avg_features))
            for rank, ch in enumerate(sorted_channels[:max(active, 1)]):
                spike_time = int((rank / max(active, 1)) * (timesteps - 1))
                spike_time = min(spike_time, timesteps - 1)
                spike_train[spike_time, ch] = 1.0
        
        return spike_train
    
    @staticmethod
    def generate_speech_command(command: str = 'yes', duration: float = 1.0,
                                 sample_rate: int = 16000) -> np.ndarray:
        """
        Generate synthetic speech-like command signals.
        Different commands have different frequency signatures.
        """
        num_samples = int(sample_rate * duration)
        t = np.arange(num_samples) / sample_rate
        
        if command == 'yes':
            # Rising frequency pattern
            freq = 200 + 300 * t / duration
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
        elif command == 'no':
            # Falling frequency pattern
            freq = 500 - 300 * t / duration
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
        elif command == 'up':
            # High frequency pulse
            audio = 0.5 * np.sin(2 * np.pi * 600 * t) * np.exp(-t / 0.2)
        elif command == 'down':
            # Low frequency pulse
            audio = 0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.3)
        elif command == 'stop':
            # Rapid alternating pattern
            audio = 0.5 * np.sin(2 * np.pi * 400 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 10 * t))
        else:
            # Default: noise
            audio = 0.3 * np.random.randn(num_samples)
        
        # Add noise
        audio += 0.02 * np.random.randn(num_samples)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        return audio


class SpikeDataset:
    """
    Generate synthetic spike-based datasets for image and audio recognition.
    Useful when real datasets (MNIST, Speech Commands) are not available.
    """
    
    @staticmethod
    def generate_image_samples(num_samples: int = 100, img_size: int = 28,
                                num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic image samples with class labels.
        Each class has a different pattern (horizontal bars, vertical bars, circles, etc.)
        
        Returns:
            images: (num_samples, img_size, img_size)
            labels: (num_samples,) integer labels
        """
        images = np.zeros((num_samples, img_size, img_size))
        labels = np.zeros(num_samples, dtype=int)
        
        samples_per_class = num_samples // num_classes
        
        for cls in range(num_classes):
            for i in range(samples_per_class):
                idx = cls * samples_per_class + i
                labels[idx] = cls
                
                if cls < 5:
                    # Horizontal bars
                    thickness = cls + 1
                    center = img_size // 2
                    y_start = center - thickness // 2
                    y_end = center + thickness // 2 + 1
                    images[idx, y_start:y_end, :] = 0.5 + 0.5 * np.random.random()
                else:
                    # Vertical bars
                    thickness = cls - 4
                    center = img_size // 2
                    x_start = center - thickness // 2
                    x_end = center + thickness // 2 + 1
                    images[idx, :, x_start:x_end] = 0.5 + 0.5 * np.random.random()
                
                # Add noise
                images[idx] += 0.05 * np.random.randn(img_size, img_size)
                images[idx] = np.clip(images[idx], 0, 1)
        
        return images, labels
    
    @staticmethod
    def generate_audio_samples(num_samples: int = 50, num_commands: int = 5) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate synthetic audio command samples.
        Commands: 'yes', 'no', 'up', 'down', 'stop'
        
        Returns:
            audio_samples: list of audio arrays
            labels: (num_samples,) integer labels
        """
        commands = ['yes', 'no', 'up', 'down', 'stop'][:num_commands]
        audio_samples = []
        labels = np.zeros(num_samples, dtype=int)
        
        samples_per_class = num_samples // num_commands
        
        for cls, cmd in enumerate(commands):
            for i in range(samples_per_class):
                idx = cls * samples_per_class + i
                labels[idx] = cls
                audio = AudioToSpikeEncoder.generate_speech_command(cmd)
                audio_samples.append(audio)
        
        return audio_samples, labels