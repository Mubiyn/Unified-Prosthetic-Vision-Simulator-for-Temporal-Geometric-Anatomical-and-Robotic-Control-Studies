#!/Users/Mubiyn/pulse-env/bin/python
"""
Biological Variation Modeling for Retinal Prostheses
===================================================

This script demonstrates how biological variations affect visual percepts
in retinal prostheses by systematically varying key patient-specific parameters
from the pulse2percept library.

Key biological parameters explored:
- rho: Spatial decay constant (how current spreads laterally)
- axlambda: Axonal decay constant (how sensitivity changes along axons)  
- atten_a/atten_n: Attenuation function parameters (current spread model)
- Patient-specific anatomical variations

Author: Biom Project
Date: Today
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import pulse2percept components
try:
    import pulse2percept as p2p
    from pulse2percept.implants import ArgusII
    from pulse2percept.models import AxonMapModel, Nanduri2012Model, ScoreboardModel
    from pulse2percept.stimuli import Stimulus, BiphasicPulse
    print("✅ pulse2percept imported successfully")
except ImportError as e:
    print(f"❌ Error importing pulse2percept: {e}")
    print("Please install pulse2percept: pip install pulse2percept")
    exit(1)

import pandas as pd
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class BiologicalVariationAnalyzer:
    """
    Comprehensive analyzer for biological variation effects in retinal prostheses
    """
    
    def __init__(self, save_results=True):
        self.save_results = save_results
        self.results = {}
        self.models = {}
        self.patient_profiles = self._create_patient_profiles()
        
        # Create output directory
        self.output_dir = Path("biological_variation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🔬 Biological Variation Analyzer Initialized")
        print(f"📁 Results will be saved to: {self.output_dir}")
        
    def _create_patient_profiles(self):
        """
        Create realistic patient profiles based on literature values
        """
        print("\n👥 Creating Patient Profiles...")
        
        # Based on Beyeler et al. 2019 and clinical literature
        profiles = {
            'young_healthy': {
                'name': 'Young Patient (Good Retinal Health)',
                'rho': 200,      # Normal spatial spread
                'axlambda': 600, # Normal axonal decay
                'description': 'Younger patient with well-preserved retinal structure'
            },
            
            'typical_patient': {
                'name': 'Typical Patient (Moderate Degeneration)',
                'rho': 300,      # Slightly increased spread
                'axlambda': 400, # Reduced axonal sensitivity
                'description': 'Average patient with moderate retinal degeneration'
            },
            
            'advanced_degeneration': {
                'name': 'Advanced Degeneration Patient',
                'rho': 500,      # High spatial spread
                'axlambda': 200, # Significant axonal loss
                'description': 'Patient with advanced retinal degeneration'
            },
            
            'elderly_patient': {
                'name': 'Elderly Patient (Age-Related Changes)',
                'rho': 400,      # Age-related increased spread
                'axlambda': 300, # Age-related axonal changes
                'description': 'Elderly patient with age-related retinal changes'
            },
            
            'focal_preservation': {
                'name': 'Focal Preservation Patient',
                'rho': 150,      # Sharp, focused responses
                'axlambda': 800, # Well-preserved axons
                'description': 'Patient with focal areas of preserved retinal function'
            }
        }
        
        print(f"✅ Created {len(profiles)} patient profiles")
        return profiles
    
    def create_models_for_profiles(self):
        """
        Create AxonMapModel instances for each patient profile
        """
        print("\n🧠 Creating Models for Each Patient Profile...")
        
        for profile_name, params in self.patient_profiles.items():
            print(f"  🔧 Building model for: {params['name']}")
            
            # Create AxonMapModel with patient-specific parameters
            model = AxonMapModel(
                rho=params['rho'],
                axlambda=params['axlambda'],
                xrange=(-8, 8),
                yrange=(-6, 6),
                xystep=0.5,
                n_axons=500,  # Reduced for faster computation
                engine='cython'  # Fast computation
            )
            
            # Build the model (expensive one-time calculation)
            print(f"    ⚙️  Building axon map...")
            model.build()
            
            self.models[profile_name] = model
            print(f"    ✅ Model built successfully")
        
        print(f"\n🎯 All {len(self.models)} models created and built!")
    
    def analyze_threshold_variations(self):
        """
        Analyze how biological variations affect stimulation thresholds
        """
        print("\n🔍 Analyzing Threshold Variations Across Patients...")
        
        # Create implant
        implant = ArgusII()
        
        # Test electrodes (representative sample)
        test_electrodes = ['A3', 'B4', 'C5', 'D6', 'E7']
        
        threshold_data = []
        
        for profile_name, model in self.models.items():
            patient_info = self.patient_profiles[profile_name]
            print(f"  🧪 Testing {patient_info['name']}...")
            
            for electrode in test_electrodes:
                # Create test stimulus
                implant.stim = {electrode: BiphasicPulse(20, 1, 0.45)}
                
                try:
                    # Find threshold
                    threshold = model.find_threshold(implant, 0.1, amp_range=(1, 200))
                    
                    threshold_data.append({
                        'Patient': patient_info['name'],
                        'Profile': profile_name,
                        'Electrode': electrode,
                        'Threshold_uA': threshold,
                        'Rho': patient_info['rho'],
                        'Axlambda': patient_info['axlambda']
                    })
                except Exception as e:
                    print(f"    ⚠️  Could not find threshold for {electrode}: {e}")
        
        self.results['thresholds'] = pd.DataFrame(threshold_data)
        print(f"  ✅ Collected threshold data for {len(threshold_data)} tests")
        
        # Plot threshold comparison
        self._plot_threshold_comparison()
    
    def _plot_threshold_comparison(self):
        """
        Create comprehensive threshold comparison plots
        """
        print("  📊 Creating threshold comparison plots...")
        
        df = self.results['thresholds']
        
        if df.empty:
            print("    ⚠️  No threshold data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Biological Variation Effects on Stimulation Thresholds', 
                     fontsize=16, fontweight='bold')
        
        # 1. Threshold by patient profile
        unique_profiles = df['Profile'].unique()
        colors = sns.color_palette("husl", len(unique_profiles))
        
        for i, profile in enumerate(unique_profiles):
            profile_data = df[df['Profile'] == profile]['Threshold_uA']
            axes[0,0].boxplot(profile_data, positions=[i], widths=0.6, 
                             patch_artist=True, 
                             boxprops=dict(facecolor=colors[i]))
        
        axes[0,0].set_title('Threshold Distribution by Patient Type')
        axes[0,0].set_xticks(range(len(unique_profiles)))
        axes[0,0].set_xticklabels(unique_profiles, rotation=45)
        axes[0,0].set_ylabel('Threshold (μA)')
        
        # 2. Threshold vs Rho
        sns.scatterplot(data=df, x='Rho', y='Threshold_uA', 
                       hue='Profile', s=80, ax=axes[0,1])
        axes[0,1].set_title('Threshold vs Spatial Decay (ρ)')
        axes[0,1].set_xlabel('Rho (μm)')
        axes[0,1].set_ylabel('Threshold (μA)')
        
        # 3. Threshold vs Axlambda
        sns.scatterplot(data=df, x='Axlambda', y='Threshold_uA', 
                       hue='Profile', s=80, ax=axes[1,0])
        axes[1,0].set_title('Threshold vs Axonal Decay (λ)')
        axes[1,0].set_xlabel('Axlambda (μm)')
        axes[1,0].set_ylabel('Threshold (μA)')
        
        # 4. Electrode-specific variations
        pivot_data = df.pivot_table(values='Threshold_uA', 
                                   index='Electrode', 
                                   columns='Profile', 
                                   aggfunc='mean')
        
        pivot_data.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Threshold Variation by Electrode Location')
        axes[1,1].set_xlabel('Electrode')
        axes[1,1].set_ylabel('Threshold (μA)')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if self.save_results:
            plt.savefig(self.output_dir / 'threshold_analysis.png', 
                       dpi=300, bbox_inches='tight')
            print(f"    💾 Saved threshold analysis plot")
        
        plt.show()
    
    def generate_comparative_percepts(self):
        """
        Generate and compare visual percepts across patient profiles
        """
        print("\n👁️  Generating Comparative Visual Percepts...")
        
        # Create implant with multi-electrode stimulus
        implant = ArgusII()
        
        # Create a simple pattern - letter 'E'
        e_pattern = {
            'A1': 50, 'A2': 50, 'A3': 50, 'A4': 50,
            'B1': 50,
            'C1': 50, 'C2': 50, 'C3': 50,
            'D1': 50,
            'E1': 50, 'E2': 50, 'E3': 50, 'E4': 50
        }
        
        percept_data = {}
        
        for profile_name, model in self.models.items():
            patient_info = self.patient_profiles[profile_name]
            print(f"  🔬 Generating percept for {patient_info['name']}...")
            
            # Set stimulation pattern
            stim_dict = {}
            for electrode, amp in e_pattern.items():
                stim_dict[electrode] = BiphasicPulse(20, amp, 0.45)
            
            implant.stim = stim_dict
            
            # Predict percept
            percept = model.predict_percept(implant)
            percept_data[profile_name] = {
                'percept': percept,
                'patient_name': patient_info['name'],
                'parameters': patient_info
            }
        
        self.results['percepts'] = percept_data
        print(f"  ✅ Generated percepts for all {len(percept_data)} patient profiles")
        
        # Plot comparative percepts
        self._plot_comparative_percepts()
    
    def _plot_comparative_percepts(self):
        """
        Create side-by-side comparison of percepts
        """
        print("  🎨 Creating comparative percept visualization...")
        
        percept_data = self.results['percepts']
        n_patients = len(percept_data)
        
        # Calculate grid size
        cols = 3
        rows = (n_patients + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        fig.suptitle('Patient-Specific Visual Percepts: Letter "E" Stimulation', 
                     fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = [axes] if n_patients == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (profile_name, data) in enumerate(percept_data.items()):
            percept = data['percept']
            patient_name = data['patient_name']
            params = data['parameters']
            
            # Plot percept - handle different data shapes
            percept_data = percept.data
            if len(percept_data.shape) == 4:  # (time, height, width, channels)
                plot_data = percept_data[0, :, :, 0]  # First time point, first channel
            elif len(percept_data.shape) == 3:  # (height, width, time) or (height, width, channels)
                plot_data = percept_data[:, :, 0]  # First time/channel
            else:  # 2D data
                plot_data = percept_data.squeeze()
            
            im = axes[idx].imshow(plot_data, cmap='hot', 
                                 extent=[percept.xdva.min(), percept.xdva.max(), 
                                        percept.ydva.min(), percept.ydva.max()], 
                                 origin='lower')
            
            axes[idx].set_title(f'{patient_name}\n'
                               f'ρ={params["rho"]}μm, λ={params["axlambda"]}μm', 
                               fontsize=11)
            axes[idx].set_xlabel('Horizontal Position (deg)')
            axes[idx].set_ylabel('Vertical Position (deg)')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(n_patients, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_results:
            plt.savefig(self.output_dir / 'comparative_percepts.png', 
                       dpi=300, bbox_inches='tight')
            print(f"    💾 Saved comparative percepts plot")
        
        plt.show()
    
    def analyze_percept_quality_metrics(self):
        """
        Quantify and analyze percept quality across patients
        """
        print("\n📏 Analyzing Percept Quality Metrics...")
        
        quality_data = []
        
        for profile_name, data in self.results['percepts'].items():
            percept = data['percept']
            patient_name = data['patient_name']
            params = data['parameters']
            
            # Calculate quality metrics
            percept_data = percept.data
            if len(percept_data.shape) == 4:  # (time, height, width, channels)
                percept_array = percept_data[0, :, :, 0]  # First time point, first channel
            elif len(percept_data.shape) == 3:  # (height, width, time) or (height, width, channels)
                percept_array = percept_data[:, :, 0]  # First time/channel
            else:  # 2D data
                percept_array = percept_data.squeeze()
            
            # 1. Peak brightness
            peak_brightness = np.max(percept_array)
            
            # 2. Spatial resolution (full width at half maximum)
            peak_coords = np.unravel_index(np.argmax(percept_array), percept_array.shape)
            half_max = peak_brightness / 2
            
            # Find FWHM in horizontal direction
            horizontal_profile = percept_array[peak_coords[0], :]
            fwhm_indices = np.where(horizontal_profile >= half_max)[0]
            fwhm_x = len(fwhm_indices) * 0.5  # Convert to degrees (0.5° step)
            
            # 3. Total activated area
            activation_threshold = peak_brightness * 0.1  # 10% of peak
            activated_area = np.sum(percept_array > activation_threshold)
            
            quality_data.append({
                'Patient': patient_name,
                'Profile': profile_name,
                'Peak_Brightness': peak_brightness,
                'FWHM_degrees': fwhm_x,
                'Activated_Area_pixels': activated_area,
                'Rho': params['rho'],
                'Axlambda': params['axlambda'],
                'Spatial_Resolution_Score': 1/fwhm_x if fwhm_x > 0 else 0,
                'Brightness_Score': peak_brightness / 100  # Normalize
            })
        
        self.results['quality_metrics'] = pd.DataFrame(quality_data)
        print(f"  ✅ Calculated quality metrics for {len(quality_data)} patients")
        
        # Plot quality analysis
        self._plot_quality_analysis()
    
    def _plot_quality_analysis(self):
        """
        Create quality metrics visualization
        """
        print("  📈 Creating quality metrics visualization...")
        
        df = self.results['quality_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Percept Quality Analysis Across Patient Profiles', 
                     fontsize=16, fontweight='bold')
        
        # 1. Brightness vs Spatial Resolution
        scatter = axes[0,0].scatter(df['FWHM_degrees'], df['Peak_Brightness'], 
                                   c=df['Rho'], s=100, cmap='viridis', alpha=0.7)
        axes[0,0].set_xlabel('FWHM (degrees) - Lower is Better')
        axes[0,0].set_ylabel('Peak Brightness')
        axes[0,0].set_title('Brightness vs Spatial Resolution')
        plt.colorbar(scatter, ax=axes[0,0], label='Rho (μm)')
        
        # Add patient labels
        for i, row in df.iterrows():
            axes[0,0].annotate(row['Profile'], 
                              (row['FWHM_degrees'], row['Peak_Brightness']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7)
        
        # 2. Parameter correlations
        sns.scatterplot(data=df, x='Rho', y='Activated_Area_pixels', 
                       hue='Profile', s=100, ax=axes[0,1])
        axes[0,1].set_title('Activated Area vs Spatial Decay (ρ)')
        axes[0,1].set_xlabel('Rho (μm)')
        axes[0,1].set_ylabel('Activated Area (pixels)')
        
        # 3. Quality scores comparison
        quality_scores = df[['Spatial_Resolution_Score', 'Brightness_Score']].values
        profile_names = df['Profile'].values
        
        x = np.arange(len(df))
        width = 0.35
        
        axes[1,0].bar(x - width/2, df['Spatial_Resolution_Score'], width, 
                     label='Spatial Resolution', alpha=0.7)
        axes[1,0].bar(x + width/2, df['Brightness_Score'], width, 
                     label='Brightness', alpha=0.7)
        
        axes[1,0].set_title('Quality Scores by Patient')
        axes[1,0].set_xlabel('Patient Profile')
        axes[1,0].set_ylabel('Quality Score (0-1)')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(profile_names, rotation=45)
        axes[1,0].legend()
        
        # 4. Parameter sensitivity
        param_data = df[['Rho', 'Axlambda', 'Peak_Brightness', 'FWHM_degrees']]
        if len(param_data) > 1:
            param_corr = param_data.corr()
            sns.heatmap(param_corr, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1,1])
            axes[1,1].set_title('Parameter Correlation Matrix')
        else:
            axes[1,1].text(0.5, 0.5, 'Need more data\nfor correlation analysis', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Parameter Correlation Matrix')
        
        plt.tight_layout()
        
        if self.save_results:
            plt.savefig(self.output_dir / 'quality_analysis.png', 
                       dpi=300, bbox_inches='tight')
            print(f"    💾 Saved quality analysis plot")
        
        plt.show()
    
    def generate_clinical_report(self):
        """
        Generate comprehensive clinical report
        """
        print("\n📋 Generating Clinical Report...")
        
        report_content = []
        report_content.append("# Biological Variation Modeling Report")
        report_content.append("## Retinal Prosthesis Patient Variability Analysis\n")
        
        # Executive Summary
        report_content.append("## Executive Summary")
        report_content.append("This analysis demonstrates significant patient-to-patient variability in retinal prosthetic vision ")
        report_content.append("due to biological differences in retinal tissue properties. Key findings:\n")
        
        if 'thresholds' in self.results and not self.results['thresholds'].empty:
            df_thresh = self.results['thresholds']
            min_thresh = df_thresh['Threshold_uA'].min()
            max_thresh = df_thresh['Threshold_uA'].max()
            report_content.append(f"- **Threshold Variability**: {min_thresh:.1f}μA to {max_thresh:.1f}μA ({max_thresh/min_thresh:.1f}x range)")
        
        if 'quality_metrics' in self.results and not self.results['quality_metrics'].empty:
            df_quality = self.results['quality_metrics']
            best_profile = df_quality.loc[df_quality['Spatial_Resolution_Score'].idxmax(), 'Patient']
            worst_profile = df_quality.loc[df_quality['Spatial_Resolution_Score'].idxmin(), 'Patient']
            report_content.append(f"- **Best Spatial Resolution**: {best_profile}")
            report_content.append(f"- **Poorest Spatial Resolution**: {worst_profile}")
        
        # Patient Profiles
        report_content.append("\n## Patient Profile Analysis")
        for profile_name, params in self.patient_profiles.items():
            report_content.append(f"\n### {params['name']}")
            report_content.append(f"- **Description**: {params['description']}")
            report_content.append(f"- **Spatial Decay (ρ)**: {params['rho']}μm")
            report_content.append(f"- **Axonal Decay (λ)**: {params['axlambda']}μm")
        
        # Clinical Implications
        report_content.append("\n## Clinical Implications")
        report_content.append("1. **Personalized Programming**: Each patient requires individual device optimization")
        report_content.append("2. **Threshold Testing**: Comprehensive threshold mapping is essential")
        report_content.append("3. **Outcome Prediction**: Biological parameters can predict visual outcomes")
        report_content.append("4. **Treatment Planning**: Patient selection and counseling should consider biological factors")
        
        # Save report
        report_text = "\n".join(report_content)
        
        if self.save_results:
            with open(self.output_dir / 'clinical_report.md', 'w') as f:
                f.write(report_text)
            
            # Also save data summaries
            if 'thresholds' in self.results:
                self.results['thresholds'].to_csv(self.output_dir / 'threshold_data.csv', index=False)
            if 'quality_metrics' in self.results:
                self.results['quality_metrics'].to_csv(self.output_dir / 'quality_metrics.csv', index=False)
        
        print("  📄 Clinical report generated")
        print(f"  💾 All data saved to {self.output_dir}")
        
        return report_text
    
    def run_complete_analysis(self):
        """
        Run the complete biological variation analysis
        """
        print("🚀 Starting Complete Biological Variation Analysis")
        print("=" * 60)
        
        try:
            # Step 1: Create models
            self.create_models_for_profiles()
            
            # Step 2: Analyze thresholds
            self.analyze_threshold_variations()
            
            # Step 3: Generate comparative percepts
            self.generate_comparative_percepts()
            
            # Step 4: Analyze quality metrics
            self.analyze_percept_quality_metrics()
            
            # Step 5: Generate report
            self.generate_clinical_report()
            
            print("\n" + "=" * 60)
            print("✅ ANALYSIS COMPLETE!")
            print(f"📁 All results saved to: {self.output_dir}")
            print("🎉 Biological variation modeling analysis finished successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            print("🔧 Check your pulse2percept installation and try again")
            import traceback
            traceback.print_exc()
            raise

def main():
    """
    Main execution function
    """
    print("🔬 Biological Variation Modeling for Retinal Prostheses")
    print("=" * 60)
    print("This analysis explores how patient-specific biological differences")
    print("affect visual percepts in retinal prostheses using pulse2percept.")
    print()
    
    # Create analyzer
    analyzer = BiologicalVariationAnalyzer(save_results=True)
    
    # Run complete analysis
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 