#!/Users/Mubiyn/pulse-env/bin/python
"""
Temporal Percept Modeling for Retinal Prostheses
===============================================

This script demonstrates how percepts change over time in retinal prostheses,
showing temporal dynamics, phosphene persistence, and dynamic scene simulation.

Option 2 from the coursework: Temporal Percept Modeling
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import pandas as pd

import pulse2percept as p2p
from pulse2percept.models import Nanduri2012Model, AxonMapModel
from pulse2percept.implants import ArgusII
from pulse2percept.stimuli import BiphasicPulse, Stimulus

class TemporalPerceptAnalyzer:
    """
    Comprehensive analyzer for temporal percept dynamics in retinal prostheses
    """
    
    def __init__(self):
        self.output_dir = Path("temporal_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🕒 Temporal Percept Analyzer Initialized")
        print(f"📁 Results will be saved to: {self.output_dir}")
    
    def create_simple_temporal_test(self):
        """Create a simple temporal test to verify functionality"""
        print("\n⚡ Creating Simple Temporal Test...")
        
        # Create temporal model
        model = Nanduri2012Model(
            xrange=(-10, 10),
            yrange=(-8, 8),
            xystep=2.0  # Coarser grid for speed
        )
        
        print("  Building temporal model...")
        model.build()
        
        # Create simple stimulus
        implant = ArgusII()
        pulse = BiphasicPulse(20, 100, 0.45, stim_dur=300)
        implant.stim = {'C5': pulse}
        
        print("  Predicting temporal percept...")
        percept = model.predict_percept(implant)
        
        print(f"  ✅ Percept shape: {percept.data.shape}")
        print(f"  ✅ Time points: {len(percept.time)}")
        print(f"  ✅ Max value: {np.max(percept.data):.3f}")
        
        # Save simple temporal plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Temporal Percept Evolution', fontsize=14, fontweight='bold')
        
        # Show 3 time points
        time_indices = [0, len(percept.time)//2, len(percept.time)-1]
        time_labels = ['Start', 'Middle', 'End']
        
        for i, (t_idx, label) in enumerate(zip(time_indices, time_labels)):
            im = axes[i].imshow(percept.data[:, :, t_idx], cmap='hot')
            axes[i].set_title(f'{label} (t={percept.time[t_idx]:.1f}ms)')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot temporal dynamics
        max_over_space = np.max(percept.data, axis=(0, 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(percept.time, max_over_space, 'b-', linewidth=2, marker='o')
        plt.xlabel('Time (ms)')
        plt.ylabel('Peak Response')
        plt.title('Temporal Response Dynamics')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'temporal_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create simple animation
        self._create_simple_animation(percept)
        
        return percept
    
    def _create_simple_animation(self, percept):
        """Create a simple animation of the temporal percept"""
        print("  🎬 Creating temporal animation...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Initialize plot
        im = ax.imshow(percept.data[:, :, 0], cmap='hot', 
                      vmin=0, vmax=np.max(percept.data))
        ax.set_title('Temporal Percept Animation')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame):
            im.set_array(percept.data[:, :, frame])
            time_text.set_text(f'Time: {percept.time[frame]:.1f} ms')
            return [im, time_text]
        
        # Create animation - sample every few frames for smoother animation
        sample_frames = np.linspace(0, len(percept.time)-1, min(20, len(percept.time))).astype(int)
        
        anim = animation.FuncAnimation(fig, animate, frames=sample_frames,
                                     interval=300, blit=True, repeat=True)
        
        # Save as GIF
        gif_path = self.output_dir / 'temporal_animation.gif'
        try:
            anim.save(gif_path, writer='pillow', fps=3)
            print(f"    💾 Animation saved: {gif_path}")
        except Exception as e:
            print(f"    ⚠️  Could not save animation: {e}")
        
        plt.close(fig)
    
    def test_different_stimuli(self):
        """Test different types of temporal stimuli"""
        print("\n🧪 Testing Different Temporal Stimuli...")
        
        # Create model
        model = Nanduri2012Model(
            xrange=(-10, 10),
            yrange=(-8, 8),
            xystep=2.0
        )
        model.build()
        
        stimuli_results = []
        
        # Test 1: Short pulse
        print("  Testing short pulse...")
        implant1 = ArgusII()
        pulse_short = BiphasicPulse(20, 100, 0.45, stim_dur=250)
        implant1.stim = {'C5': pulse_short}
        
        percept1 = model.predict_percept(implant1)
        max_response1 = np.max(percept1.data)
        duration1 = percept1.time[-1] - percept1.time[0]
        
        stimuli_results.append({
            'Stimulus': 'Short Pulse',
            'Duration_ms': duration1,
            'Max_Response': max_response1,
            'Time_Points': len(percept1.time)
        })
        
        # Test 2: Long pulse
        print("  Testing long pulse...")
        implant2 = ArgusII()
        pulse_long = BiphasicPulse(20, 100, 0.45, stim_dur=500)
        implant2.stim = {'C5': pulse_long}
        
        percept2 = model.predict_percept(implant2)
        max_response2 = np.max(percept2.data)
        duration2 = percept2.time[-1] - percept2.time[0]
        
        stimuli_results.append({
            'Stimulus': 'Long Pulse',
            'Duration_ms': duration2,
            'Max_Response': max_response2,
            'Time_Points': len(percept2.time)
        })
        
        # Test 3: High amplitude
        print("  Testing high amplitude...")
        implant3 = ArgusII()
        pulse_high = BiphasicPulse(20, 150, 0.45, stim_dur=350)
        implant3.stim = {'C5': pulse_high}
        
        percept3 = model.predict_percept(implant3)
        max_response3 = np.max(percept3.data)
        duration3 = percept3.time[-1] - percept3.time[0]
        
        stimuli_results.append({
            'Stimulus': 'High Amplitude',
            'Duration_ms': duration3,
            'Max_Response': max_response3,
            'Time_Points': len(percept3.time)
        })
        
        # Create comparison plots
        self._plot_stimuli_comparison([percept1, percept2, percept3], stimuli_results)
        
        # Save results
        df = pd.DataFrame(stimuli_results)
        df.to_csv(self.output_dir / 'stimuli_comparison.csv', index=False)
        print(f"  📊 Results saved to stimuli_comparison.csv")
        
        return stimuli_results
    
    def _plot_stimuli_comparison(self, percepts, results):
        """Plot comparison of different stimuli"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Temporal Stimuli Comparison', fontsize=14, fontweight='bold')
        
        colors = ['blue', 'orange', 'green']
        
        # Time courses
        for i, (percept, result) in enumerate(zip(percepts, results)):
            max_over_space = np.max(percept.data, axis=(0, 1))
            axes[0, 0].plot(percept.time, max_over_space, 
                           label=result['Stimulus'], 
                           color=colors[i], linewidth=2)
        
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Peak Response')
        axes[0, 0].set_title('Temporal Response Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Peak responses comparison
        stimuli_names = [r['Stimulus'] for r in results]
        max_responses = [r['Max_Response'] for r in results]
        
        bars = axes[0, 1].bar(stimuli_names, max_responses, 
                             color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Peak Response')
        axes[0, 1].set_title('Peak Response Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, max_responses):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{val:.2f}', ha='center', va='bottom')
        
        # Duration comparison
        durations = [r['Duration_ms'] for r in results]
        bars2 = axes[1, 0].bar(stimuli_names, durations, 
                              color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Duration (ms)')
        axes[1, 0].set_title('Stimulus Duration Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars2, durations):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           f'{val:.0f}', ha='center', va='bottom')
        
        # Spatial activation at peak time
        for i, percept in enumerate(percepts):
            # Find peak time
            max_over_space = np.max(percept.data, axis=(0, 1))
            peak_time_idx = np.argmax(max_over_space)
            
            # Show spatial pattern at peak
            if i == 0:  # Show first stimulus as example
                im = axes[1, 1].imshow(percept.data[:, :, peak_time_idx], cmap='hot')
                axes[1, 1].set_title(f'{results[i]["Stimulus"]} - Spatial Pattern at Peak')
                plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stimuli_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_simple_report(self):
        """Generate a simple temporal analysis report"""
        print("\n📋 Generating Temporal Analysis Report...")
        
        report = []
        report.append("# Temporal Percept Modeling Analysis")
        report.append("## Retinal Prosthesis Temporal Dynamics\n")
        
        report.append("## Overview")
        report.append("This analysis demonstrates temporal dynamics in retinal prosthetic vision using the Nanduri2012 temporal model.")
        report.append("Key findings show how percepts evolve over time with different stimulation parameters.\n")
        
        report.append("## Analysis Components")
        report.append("1. **Basic Temporal Response**: Single pulse temporal evolution")
        report.append("2. **Stimulus Comparison**: Different pulse durations and amplitudes")
        report.append("3. **Temporal Animation**: Visual demonstration of percept evolution")
        report.append("4. **Quantitative Metrics**: Peak response, duration, and dynamics\n")
        
        # Load and include results if available
        try:
            df = pd.read_csv(self.output_dir / 'stimuli_comparison.csv')
            report.append("## Stimulus Comparison Results")
            for _, row in df.iterrows():
                report.append(f"\n### {row['Stimulus']}")
                report.append(f"- **Duration**: {row['Duration_ms']:.1f}ms")
                report.append(f"- **Peak Response**: {row['Max_Response']:.3f}")
                report.append(f"- **Time Points**: {row['Time_Points']}")
        except:
            pass
        
        report.append("\n## Clinical Relevance")
        report.append("- **Temporal persistence**: Percepts persist beyond stimulus duration")
        report.append("- **Amplitude effects**: Higher amplitudes produce stronger, longer responses")
        report.append("- **Duration effects**: Longer stimuli create sustained percepts")
        report.append("- **Optimization potential**: Temporal parameters can be tuned for better vision")
        
        report.append("\n## Technical Implementation")
        report.append("- **Model**: Nanduri2012 temporal model with phosphene dynamics")
        report.append("- **Stimuli**: Biphasic pulses with varying parameters")
        report.append("- **Analysis**: Spatial-temporal percept evolution")
        report.append("- **Visualization**: Static plots and animated sequences")
        
        report_text = "\n".join(report)
        
        with open(self.output_dir / 'temporal_analysis_report.md', 'w') as f:
            f.write(report_text)
        
        print(f"  📄 Report saved to {self.output_dir}/temporal_analysis_report.md")
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete temporal analysis"""
        print("🚀 Starting Temporal Percept Modeling Analysis")
        print("=" * 60)
        
        try:
            # Step 1: Simple temporal test
            self.create_simple_temporal_test()
            
            # Step 2: Different stimuli comparison
            self.test_different_stimuli()
            
            # Step 3: Generate report
            self.generate_simple_report()
            
            print("\n" + "=" * 60)
            print("✅ TEMPORAL ANALYSIS COMPLETE!")
            print(f"📁 Results saved to: {self.output_dir}")
            print("🎬 Check the GIF animation for temporal dynamics!")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Run the temporal percept analysis"""
    print("🕒 Temporal Percept Modeling for Retinal Prostheses")
    print("=" * 60)
    print("This analysis shows how retinal prosthetic percepts evolve over time")
    print()
    
    analyzer = TemporalPerceptAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 