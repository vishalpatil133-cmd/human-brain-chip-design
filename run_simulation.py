"""
Neuromorphic Chip Simulation - Main Entry Point
Demonstrates: LIF neuron dynamics, energy efficiency vs dense MAC,
image recognition, audio recognition, and cross-modal integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

from neuron_model import LIFNeuron, LIFParams, EnergyParams, LIFLayer
from crossbar_simulator import CrossbarCore, generate_poisson_spike_trains, run_sparsity_sweep, TemporalPatternClassifier
from encoding import ImageToSpikeEncoder, AudioToSpikeEncoder, SpikeDataset
from recognition_model import ImageRecognitionModel, AudioRecognitionModel, SNNClassifier


OUTPUT_DIR = "output"


def ensure_output_dir():
    """Create output directory for plots and results"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return OUTPUT_DIR


def demo_single_neuron():
    """
    Demo 1: Single LIF neuron behavior.
    Shows membrane potential dynamics, spiking, and refractory period.
    """
    print("=" * 60)
    print("DEMO 1: Single LIF Neuron Dynamics")
    print("=" * 60)
    
    neuron = LIFNeuron()
    timesteps = 500
    membrane_potential = np.zeros(timesteps)
    spike_events = np.zeros(timesteps)
    
    # Constant input current with a pulse
    for t in range(timesteps):
        if 100 <= t <= 300:
            input_current = 0.2
        else:
            input_current = 0.0
        spiked = neuron.step(input_current)
        membrane_potential[t] = neuron.v_membrane
        spike_events[t] = 1.0 if spiked else 0.0
    
    stats = neuron.get_stats()
    print(f"  Total spikes: {stats['spike_count']}")
    print(f"  Leak operations: {stats['leak_ops']}")
    print(f"  Energy consumed: {stats['energy_pj']:.2f} pJ")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    ax1.plot(membrane_potential, color='blue', linewidth=1.5)
    ax1.axhline(y=neuron.params.v_threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({neuron.params.v_threshold})')
    ax1.set_ylabel('Membrane Potential (V)')
    ax1.set_title('LIF Neuron Membrane Potential Dynamics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.stem(range(timesteps), spike_events, basefmt=' ', markerfmt='ro', linefmt='r-')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Spike Event')
    ax2.set_title('Output Spike Train')
    ax2.set_ylim(-0.1, 1.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_single_neuron_dynamics.png"), dpi=150)
    print(f"  Plot saved: output/01_single_neuron_dynamics.png")
    plt.close()
    
    return stats


def demo_energy_comparison():
    """
    Demo 2: Energy comparison between neuromorphic and traditional GPU/CPU.
    Shows how sparsity directly affects energy savings.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Energy Efficiency Comparison (Neuromorphic vs Dense MAC)")
    print("=" * 60)
    
    # Run sparsity sweep
    results = run_sparsity_sweep(num_trials=3)
    
    firing_rates = []
    efficiency_ratios = []
    sparsities = []
    neuro_energies = []
    dense_energies = []
    
    for key, data in sorted(results.items()):
        firing_rates.append(data['firing_rate'])
        efficiency_ratios.append(data['efficiency_ratio'])
        sparsities.append(data['sparsity'] * 100)
        neuro_energies.append(data['neuromorphic_energy_pj'] / 1e3)  # Convert to nJ
        dense_energies.append(data['dense_energy_pj'] / 1e3)
        print(f"  Firing rate: {data['firing_rate']:.2f} | "
              f"Sparsity: {data['sparsity']*100:.1f}% | "
              f"Efficiency ratio: {data['efficiency_ratio']:.1f}x | "
              f"Neuro: {data['neuromorphic_energy_pj']:.0f} pJ | "
              f"Dense: {data['dense_energy_pj']:.0f} pJ")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Energy comparison bar chart
    x = np.arange(len(firing_rates))
    width = 0.35
    ax1.bar(x - width/2, neuro_energies, width, label='Neuromorphic (Event-Driven)', color='green', alpha=0.8)
    ax1.bar(x + width/2, dense_energies, width, label='Dense GPU/CPU (MAC)', color='red', alpha=0.6)
    ax1.set_xlabel('Input Firing Rate')
    ax1.set_ylabel('Energy (nJ)')
    ax1.set_title('Energy Consumption: Neuromorphic vs Dense')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r:.0%}' for r in firing_rates])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Efficiency ratio
    ax2.plot(firing_rates, efficiency_ratios, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Input Firing Rate')
    ax2.set_ylabel('Energy Efficiency Ratio (Dense / Neuromorphic)')
    ax2.set_title('Energy Efficiency vs Input Activity')
    ax2.grid(True, alpha=0.3)
    
    # Annotate
    for i, (r, e) in enumerate(zip(firing_rates, efficiency_ratios)):
        ax2.annotate(f'{e:.1f}x', (r, e), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_energy_comparison.png"), dpi=150)
    print(f"  Plot saved: output/02_energy_comparison.png")
    plt.close()
    
    return results


def demo_temporal_pattern_classification():
    """
    Demo 3: Temporal pattern classification using the neuromorphic core.
    Core identifies specific spike timing patterns.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Temporal Pattern Classification")
    print("=" * 60)
    
    classifier = TemporalPatternClassifier()
    
    # Run benchmark
    benchmark = classifier.run_benchmark(num_samples=15, timesteps=100)
    print(f"  Classification accuracy: {benchmark['accuracy']:.2%}")
    print(f"  Correct: {benchmark['correct']}/{benchmark['total']}")
    
    # Show energy comparison for a single classification
    spike_train = classifier.generate_pattern(0, 100)
    core = classifier.core
    core.run_timesteps(spike_train)
    comp = core.get_energy_comparison()
    print(f"  Energy per classification:")
    print(f"    Neuromorphic: {comp['neuromorphic_energy_pj']:.1f} pJ")
    print(f"    Dense equivalent: {comp['dense_mac_energy_pj']:.1f} pJ")
    print(f"    Savings: {comp['energy_savings_pct']:.1f}%")
    print(f"    Efficiency ratio: {comp['energy_efficiency_ratio']:.1f}x")
    
    # Generate output spikes for visualization
    output_spikes = core.activity_trace
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    
    patterns = ['Early Burst', 'Late Burst', 'Rhythmic', 'Noise']
    colors = ['blue', 'orange', 'green', 'red']
    
    for i, ax in enumerate(axes[:4]):
        train = classifier.generate_pattern(i, 100)
        core.reset()
        core.run_timesteps(train)
        
        # Raster plot
        for t in range(100):
            active = np.where(train[t] > 0)[0]
            ax.scatter([t] * len(active), active, s=1, c=colors[i], alpha=0.5)
        
        ax.set_ylabel(f'{patterns[i]}\nInput Neuron')
        ax.set_ylim(0, 64)
        ax.grid(True, alpha=0.2)
        ax.set_title(f'{patterns[i]} Pattern - Input Raster')
    
    # Energy savings plot
    rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    savings = []
    for rate in rates:
        core_test = CrossbarCore(num_inputs=64, num_neurons=4)
        spikes = generate_poisson_spike_trains(64, 200, rate)
        core_test.run_timesteps(spikes)
        comp = core_test.get_energy_comparison()
        savings.append(comp['energy_savings_pct'])
    
    axes[4].plot(rates, savings, 'go-', linewidth=2, markersize=8)
    axes[4].axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% savings threshold')
    axes[4].set_xlabel('Input Firing Rate')
    axes[4].set_ylabel('Energy Savings (%)')
    axes[4].set_title('Energy Savings vs Input Sparsity')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_pattern_classification.png"), dpi=150)
    print(f"  Plot saved: output/03_pattern_classification.png")
    plt.close()
    
    return benchmark


def demo_image_recognition():
    """
    Demo 4: Image recognition on neuromorphic chip.
    Uses synthetic image dataset with spike encoding.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Image Recognition on Neuromorphic Chip")
    print("=" * 60)
    
    # Generate synthetic dataset
    images, labels = SpikeDataset.generate_image_samples(num_samples=50, img_size=28, num_classes=5)
    print(f"  Generated {len(images)} synthetic image samples with {5} classes")
    
    # Split into train/test
    split = int(0.8 * len(images))
    train_images, test_images = images[:split], images[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    
    # Initialize image recognition model
    model = ImageRecognitionModel(img_size=28, hidden_size=64, num_classes=5, timesteps=50)
    
    # Train
    print("  Training SNN...")
    accuracies = model.train_on_dataset(train_images, train_labels, epochs=5)
    print(f"  Final training accuracy: {accuracies[-1]:.2%}")
    
    # Evaluate
    eval_results = model.evaluate(test_images, test_labels)
    print(f"  Test accuracy: {eval_results['accuracy']:.2%}")
    print(f"  Avg neuromorphic energy: {eval_results['avg_neuro_energy_pj']:.1f} pJ")
    print(f"  Avg dense equivalent: {eval_results['avg_dense_energy_pj']:.1f} pJ")
    print(f"  Energy efficiency: {eval_results['energy_efficiency_ratio']:.1f}x")
    print(f"  Energy savings: {eval_results['energy_savings_pct']:.1f}%")
    
    # Visualize sample classifications
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(10):
        ax = axes[i // 5, i % 5]
        img = test_images[i] if i < len(test_images) else train_images[i % len(train_images)]
        true_label = test_labels[i] if i < len(test_images) else train_labels[i % len(train_images)]
        
        # Classify
        pred, spikes, _ = model.classify_image(img)
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}\nPred: {pred}\nSpikes: {int(spikes[pred])}', fontsize=9)
        ax.axis('off')
    
    plt.suptitle('Image Recognition Results - Neuromorphic SNN', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_image_recognition.png"), dpi=150)
    print(f"  Plot saved: output/04_image_recognition.png")
    plt.close()
    
    return eval_results


def demo_audio_recognition():
    """
    Demo 5: Audio/speech command recognition on neuromorphic chip.
    Uses synthetic speech commands: 'yes', 'no', 'up', 'down', 'stop'.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Audio/Speech Command Recognition on Neuromorphic Chip")
    print("=" * 60)
    
    # Generate synthetic audio dataset
    audio_samples, labels = SpikeDataset.generate_audio_samples(num_samples=40, num_commands=5)
    command_names = ['yes', 'no', 'up', 'down', 'stop']
    print(f"  Generated {len(audio_samples)} audio samples with {5} commands")
    
    # Split
    split = int(0.8 * len(audio_samples))
    train_audio, test_audio = audio_samples[:split], audio_samples[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    
    # Initialize audio recognition model
    model = AudioRecognitionModel(num_features=8, hidden_size=32, num_commands=5, timesteps=50)
    
    # Train
    print("  Training SNN on audio commands...")
    accuracies = model.train_on_dataset(train_audio, train_labels, epochs=5)
    print(f"  Final training accuracy: {accuracies[-1]:.2%}")
    
    # Evaluate
    eval_results = model.evaluate(test_audio, test_labels)
    print(f"  Test accuracy: {eval_results['accuracy']:.2%}")
    print(f"  Avg neuromorphic energy: {eval_results['avg_neuro_energy_pj']:.1f} pJ")
    print(f"  Avg dense equivalent: {eval_results['avg_dense_energy_pj']:.1f} pJ")
    print(f"  Energy efficiency: {eval_results['energy_efficiency_ratio']:.1f}x")
    print(f"  Energy savings: {eval_results['energy_savings_pct']:.1f}%")
    
    # Visualize audio waveforms and encoded spikes
    fig, axes = plt.subplots(5, 2, figsize=(14, 12))
    
    for i, cmd in enumerate(command_names[:5]):
        audio = AudioToSpikeEncoder.generate_speech_command(cmd)
        spikes = AudioToSpikeEncoder.encode_audio(audio, 16000, timesteps=100, method='poisson', num_channels=8)
        
        # Waveform
        t = np.arange(len(audio)) / 16000
        axes[i, 0].plot(t[:1600], audio[:1600], linewidth=0.5, color='blue')
        axes[i, 0].set_title(f'Command: "{cmd}" - Audio Waveform')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Spike raster
        for ch in range(8):
            spike_times = np.where(spikes[:, ch] > 0)[0]
            axes[i, 1].scatter(spike_times, [ch] * len(spike_times), s=2, c='black', alpha=0.6)
        
        axes[i, 1].set_title(f'Command: "{cmd}" - Spike Encoding (8 Channels)')
        axes[i, 1].set_xlabel('Timestep')
        axes[i, 1].set_ylabel('Channel')
        axes[i, 1].set_ylim(-0.5, 7.5)
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_audio_recognition.png"), dpi=150)
    print(f"  Plot saved: output/05_audio_recognition.png")
    plt.close()
    
    return eval_results


def demo_crossbar_sweep():
    """
    Demo 6: Crossbar scaling analysis.
    Shows how energy efficiency scales with core size.
    """
    print("\n" + "=" * 60)
    print("DEMO 6: Crossbar Scaling Analysis")
    print("=" * 60)
    
    core_sizes = [32, 64, 128, 256, 512]
    efficiencies = []
    
    for size in core_sizes:
        core = CrossbarCore(num_inputs=size, num_neurons=size)
        spikes = generate_poisson_spike_trains(size, 100, 0.1)
        core.run_timesteps(spikes)
        comp = core.get_energy_comparison()
        efficiencies.append(comp['energy_efficiency_ratio'])
        print(f"  Core {size}x{size}: {comp['energy_efficiency_ratio']:.1f}x efficiency | "
              f"{comp['energy_savings_pct']:.1f}% savings")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(core_sizes, efficiencies, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Crossbar Size (N x N)')
    ax.set_ylabel('Energy Efficiency Ratio (Dense / Neuromorphic)')
    ax.set_title('Neuromorphic Energy Efficiency vs Core Size\n(Firing Rate: 10%)')
    ax.grid(True, alpha=0.3)
    
    for size, eff in zip(core_sizes, efficiencies):
        ax.annotate(f'{eff:.1f}x', (size, eff), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_crossbar_scaling.png"), dpi=150)
    print(f"  Plot saved: output/06_crossbar_scaling.png")
    plt.close()
    
    return dict(zip(core_sizes, efficiencies))


def demo_membrane_raster():
    """
    Demo 7: Multi-neuron raster plot showing population activity and energy over time.
    """
    print("\n" + "=" * 60)
    print("DEMO 7: Population Activity Raster & Energy Timeline")
    print("=" * 60)
    
    core = CrossbarCore(num_inputs=100, num_neurons=50)
    spikes = generate_poisson_spike_trains(100, 300, 0.08)
    output_spikes = core.run_timesteps(spikes)
    comp = core.get_energy_comparison()
    
    print(f"  Total output spikes: {comp['total_spikes']}")
    print(f"  Total energy: {comp['neuromorphic_energy_pj']:.0f} pJ")
    print(f"  Efficiency: {comp['energy_efficiency_ratio']:.1f}x")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Input raster
    for t in range(300):
        active = np.where(spikes[t] > 0)[0]
        axes[0].scatter([t] * len(active), active, s=1, c='blue', alpha=0.3)
    axes[0].set_ylabel('Input Neuron')
    axes[0].set_title(f'Input Spike Raster (Firing Rate: 8%)')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.2)
    
    # Output raster
    for t in range(300):
        active = np.where(output_spikes[t] > 0)[0]
        axes[1].scatter([t] * len(active), active, s=2, c='red', alpha=0.5)
    axes[1].set_ylabel('Output Neuron')
    axes[1].set_title(f'Output Spike Raster ({comp["total_spikes"]} spikes)')
    axes[1].set_ylim(0, 50)
    axes[1].grid(True, alpha=0.2)
    
    # Energy accumulation over time
    cumulative_energy = np.cumsum(core.energy_trace) / 1000  # Convert to nJ
    axes[2].plot(cumulative_energy, color='green', linewidth=2)
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('Cumulative Energy (nJ)')
    axes[2].set_title('Energy Accumulation Over Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_raster_energy_timeline.png"), dpi=150)
    print(f"  Plot saved: output/07_raster_energy_timeline.png")
    plt.close()


def demo_cross_modal():
    """
    Demo 8: Cross-modal integration (image + audio) demonstration.
    """
    print("\n" + "=" * 60)
    print("DEMO 8: Cross-Modal (Image + Audio) Integration Demo")
    print("=" * 60)
    
    from recognition_model import CrossModalIntegration
    
    # Create cross-modal model
    cm_model = CrossModalIntegration(img_size=14, audio_features=8, 
                                      hidden_size=32, num_classes=3, timesteps=50)
    
    # Generate synthetic image
    images, _ = SpikeDataset.generate_image_samples(num_samples=5, img_size=14, num_classes=3)
    sample_image = images[0]
    
    # Generate synthetic audio
    sample_audio = AudioToSpikeEncoder.generate_speech_command('yes')
    
    # Classify
    prediction, energy = cm_model.classify_multimodal(sample_image, sample_audio)
    
    print(f"  Multi-modal prediction: Class {prediction}")
    print(f"  Neuromorphic energy: {energy['neuromorphic_energy_pj']:.1f} pJ")
    print(f"  Dense equivalent: {energy['dense_mac_energy_pj']:.1f} pJ")
    print(f"  Efficiency: {energy['efficiency_ratio']:.1f}x")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(sample_image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    t = np.arange(len(sample_audio[:2000])) / 16000
    axes[1].plot(t, sample_audio[:2000], color='blue', linewidth=0.5)
    axes[1].set_title('Input Audio Waveform')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    
    metrics = ['Neuro\nEnergy', 'Dense\nEnergy', 'Efficiency\nRatio']
    values = [energy['neuromorphic_energy_pj'], 
              energy['dense_mac_energy_pj'],
              energy['efficiency_ratio'] / 10]  # Scaled for visibility
    colors_bar = ['green', 'red', 'blue']
    axes[2].bar(metrics, values, color=colors_bar, alpha=0.7)
    axes[2].set_title(f'Energy Analysis\n(Prediction: Class {prediction})')
    axes[2].set_ylabel('pJ / Scaled Ratio')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_cross_modal_integration.png"), dpi=150)
    print(f"  Plot saved: output/08_cross_modal_integration.png")
    plt.close()


def print_summary(results: dict):
    """Print comprehensive summary"""
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE - RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n📊 ENERGY EFFICIENCY KEY FINDINGS:")
    print("-" * 40)
    for rate, data in sorted(results.get('sparsity_sweep', {}).items()):
        print(f"  Firing Rate {data['firing_rate']:.0%}: {data['efficiency_ratio']:.1f}x more efficient")
    
    print(f"\n🧠 PATTERN CLASSIFICATION:")
    print("-" * 40)
    pc = results.get('pattern_classification', {})
    if pc:
        print(f"  Accuracy: {pc.get('accuracy', 0):.2%}")
    
    print(f"\n🖼️  IMAGE RECOGNITION:")
    print("-" * 40)
    ir = results.get('image_recognition', {})
    if ir:
        print(f"  Test Accuracy: {ir.get('accuracy', 0):.2%}")
        print(f"  Energy Efficiency: {ir.get('energy_efficiency_ratio', 0):.1f}x")
        print(f"  Energy Savings: {ir.get('energy_savings_pct', 0):.1f}%")
    
    print(f"\n🔊 AUDIO RECOGNITION:")
    print("-" * 40)
    ar = results.get('audio_recognition', {})
    if ar:
        print(f"  Test Accuracy: {ar.get('accuracy', 0):.2%}")
        print(f"  Energy Efficiency: {ar.get('energy_efficiency_ratio', 0):.1f}x")
        print(f"  Energy Savings: {ar.get('energy_savings_pct', 0):.1f}%")
    
    print(f"\n⚡ CROSSBAR SCALING:")
    print("-" * 40)
    cs = results.get('crossbar_scaling', {})
    if cs:
        for size, eff in cs.items():
            print(f"  {size}x{size}: {eff:.1f}x efficiency")


def main():
    """Main entry point - run all demonstrations"""
    print("=" * 60)
    print("NEUROMORPHIC CHIP DESIGN & ENERGY EFFICIENCY SIMULATION")
    print("=" * 60)
    print("Model: 256-Neuron LIF Digital Crossbar (8-bit weights)")
    print("Comparison: Event-Driven vs Dense GPU/CPU MAC operations")
    print("Tasks: Image Recognition + Audio/Speech Recognition + Cross-Modal")
    print("=" * 60)
    
    ensure_output_dir()
    
    results = {}
    
    try:
        results['single_neuron'] = demo_single_neuron()
    except Exception as e:
        print(f"  Error in Demo 1: {e}")
    
    try:
        results['sparsity_sweep'] = demo_energy_comparison()
    except Exception as e:
        print(f"  Error in Demo 2: {e}")
    
    try:
        results['pattern_classification'] = demo_temporal_pattern_classification()
    except Exception as e:
        print(f"  Error in Demo 3: {e}")
    
    try:
        results['image_recognition'] = demo_image_recognition()
    except Exception as e:
        print(f"  Error in Demo 4: {e}")
    
    try:
        results['audio_recognition'] = demo_audio_recognition()
    except Exception as e:
        print(f"  Error in Demo 5: {e}")
    
    try:
        results['crossbar_scaling'] = demo_crossbar_sweep()
    except Exception as e:
        print(f"  Error in Demo 6: {e}")
    
    try:
        demo_membrane_raster()
    except Exception as e:
        print(f"  Error in Demo 7: {e}")
    
    try:
        demo_cross_modal()
    except Exception as e:
        print(f"  Error in Demo 8: {e}")
    
    print_summary(results)
    
    print("\n" + "=" * 60)
    print("Simulation Complete! All plots saved to output/")
    print("=" * 60)


if __name__ == "__main__":
    main()