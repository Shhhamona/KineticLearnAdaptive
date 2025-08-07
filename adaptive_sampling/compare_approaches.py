#!/usr/bin/env python3
"""
Approach Comparison Script

This script compares results from traditional ML and adaptive sampling approaches,
providing comprehensive cross-approach analysis and visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

class ApproachComparison:
    """Compare traditional and adaptive approaches."""
    
    def __init__(self):
        self.results_dir = Path('results')
        self.comparison_dir = Path('results') / 'comparison' / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Comparison results will be saved to: {self.comparison_dir}")
    
    def find_result_files(self):
        """Find all summary.csv files from both approaches."""
        traditional_files = list(self.results_dir.glob('traditional/**/summary.csv'))
        adaptive_files = list(self.results_dir.glob('adaptive/**/summary.csv'))
        
        print(f"Found {len(traditional_files)} traditional result files")
        print(f"Found {len(adaptive_files)} adaptive result files")
        
        return traditional_files, adaptive_files
    
    def load_and_combine_results(self, traditional_files, adaptive_files):
        """Load and combine results from both approaches."""
        all_results = []
        
        # Load traditional results
        for file_path in traditional_files:
            try:
                df = pd.read_csv(file_path)
                df['result_file'] = str(file_path)
                df['simulator'] = file_path.parts[-4]  # Extract simulator name from path
                df['chemistry'] = file_path.parts[-3]  # Extract chemistry name from path
                all_results.append(df)
                print(f"   Loaded: {file_path}")
            except Exception as e:
                print(f"   Error loading {file_path}: {e}")
        
        # Load adaptive results
        for file_path in adaptive_files:
            try:
                df = pd.read_csv(file_path)
                df['result_file'] = str(file_path)
                df['simulator'] = file_path.parts[-4]  # Extract simulator name from path
                df['chemistry'] = file_path.parts[-3]  # Extract chemistry name from path
                all_results.append(df)
                print(f"   Loaded: {file_path}")
            except Exception as e:
                print(f"   Error loading {file_path}: {e}")
        
        if not all_results:
            raise ValueError("No result files could be loaded!")
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    
    def standardize_metrics(self, df):
        """Standardize metrics across both approaches."""
        # Fill missing approach values based on file path
        for idx, row in df.iterrows():
            if pd.isna(row['approach']) or row['approach'] == '':
                if 'traditional' in row['result_file']:
                    df.at[idx, 'approach'] = 'traditional'
                elif 'adaptive' in row['result_file']:
                    df.at[idx, 'approach'] = 'adaptive'
        
        # Create standardized columns
        df['primary_rmse'] = np.nan
        df['primary_r2'] = np.nan
        
        # For traditional approach: use test metrics
        traditional_mask = df['approach'] == 'traditional'
        if 'test_rmse' in df.columns:
            df.loc[traditional_mask, 'primary_rmse'] = df.loc[traditional_mask, 'test_rmse']
        if 'test_r2' in df.columns:
            df.loc[traditional_mask, 'primary_r2'] = df.loc[traditional_mask, 'test_r2']
        
        # For adaptive approach: use final metrics
        adaptive_mask = df['approach'] == 'adaptive'
        if 'final_rmse' in df.columns:
            df.loc[adaptive_mask, 'primary_rmse'] = df.loc[adaptive_mask, 'final_rmse']
        if 'final_r2_score' in df.columns:
            df.loc[adaptive_mask, 'primary_r2'] = df.loc[adaptive_mask, 'final_r2_score']
        
        return df
    
    def generate_comparison_plots(self, df):
        """Generate comprehensive comparison plots."""
        print("\n📈 Generating comparison plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. RMSE comparison by approach and model
        self.plot_rmse_comparison(df)
        
        # 2. R² comparison by approach and model
        self.plot_r2_comparison(df)
        
        # 3. Efficiency comparison (performance vs simulations)
        self.plot_efficiency_comparison(df)
        
        # 4. Model-specific comparison
        self.plot_model_specific_comparison(df)
    
    def plot_rmse_comparison(self, df):
        """Plot RMSE comparison between approaches."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: RMSE by approach and model
        ax = axes[0]
        sns.barplot(data=df, x='model_type', y='primary_rmse', hue='approach', ax=ax)
        ax.set_ylabel('RMSE (Primary Metric)')
        ax.set_xlabel('Model Type')
        ax.set_title('RMSE Comparison by Approach')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Plot 2: RMSE distribution
        ax = axes[1]
        for approach in df['approach'].unique():
            subset = df[df['approach'] == approach]
            ax.hist(subset['primary_rmse'], alpha=0.7, label=approach, bins=10)
        ax.set_xlabel('RMSE (Primary Metric)')
        ax.set_ylabel('Frequency')
        ax.set_title('RMSE Distribution by Approach')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.comparison_dir / 'rmse_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path}")
    
    def plot_r2_comparison(self, df):
        """Plot R² comparison between approaches."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: R² by approach and model
        ax = axes[0]
        sns.barplot(data=df, x='model_type', y='primary_r2', hue='approach', ax=ax)
        ax.set_ylabel('R² Score (Primary Metric)')
        ax.set_xlabel('Model Type')
        ax.set_title('R² Score Comparison by Approach')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Plot 2: R² distribution
        ax = axes[1]
        for approach in df['approach'].unique():
            subset = df[df['approach'] == approach]
            ax.hist(subset['primary_r2'], alpha=0.7, label=approach, bins=10)
        ax.set_xlabel('R² Score (Primary Metric)')
        ax.set_ylabel('Frequency')
        ax.set_title('R² Score Distribution by Approach')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.comparison_dir / 'r2_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path}")
    
    def plot_efficiency_comparison(self, df):
        """Plot efficiency comparison (performance vs cost)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot with different markers for each approach
        for approach in df['approach'].unique():
            subset = df[df['approach'] == approach]
            marker = 'o' if approach == 'traditional' else '^'
            ax.scatter(subset['total_simulations'], subset['primary_r2'], 
                      alpha=0.7, label=approach, s=100, marker=marker)
        
        ax.set_xlabel('Total Simulations', fontsize=12)
        ax.set_ylabel('R² Score (Primary Metric)', fontsize=12)
        ax.set_title('Efficiency: Performance vs Simulation Cost', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Add trend lines
        for approach in df['approach'].unique():
            subset = df[df['approach'] == approach]
            if len(subset) > 1:
                z = np.polyfit(subset['total_simulations'], subset['primary_r2'], 1)
                p = np.poly1d(z)
                ax.plot(subset['total_simulations'], p(subset['total_simulations']), 
                       linestyle='--', alpha=0.8)
        
        plt.tight_layout()
        plot_path = self.comparison_dir / 'efficiency_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path}")
    
    def plot_model_specific_comparison(self, df):
        """Plot model-specific detailed comparison."""
        models = df['model_type'].unique()
        fig, axes = plt.subplots(len(models), 2, figsize=(15, 6*len(models)))
        
        if len(models) == 1:
            axes = axes.reshape(1, -1)
        
        for i, model in enumerate(models):
            model_data = df[df['model_type'] == model]
            
            # RMSE comparison for this model
            ax = axes[i, 0]
            sns.barplot(data=model_data, x='approach', y='primary_rmse', ax=ax)
            ax.set_ylabel('RMSE')
            ax.set_title(f'{model.title()} - RMSE Comparison')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # R² comparison for this model
            ax = axes[i, 1]
            sns.barplot(data=model_data, x='approach', y='primary_r2', ax=ax)
            ax.set_ylabel('R² Score')
            ax.set_title(f'{model.title()} - R² Score Comparison')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.comparison_dir / 'model_specific_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path}")
    
    def generate_summary_report(self, df):
        """Generate a comprehensive summary report."""
        print("\n📝 Generating comparison report...")
        
        report_file = self.comparison_dir / 'comparison_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Approach Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overall Statistics\n\n")
            
            # Summary statistics by approach
            for approach in df['approach'].unique():
                if pd.isna(approach):
                    continue
                subset = df[df['approach'] == approach]
                f.write(f"### {approach.title()} Approach\n")
                f.write(f"- Number of runs: {len(subset)}\n")
                f.write(f"- Mean RMSE: {subset['primary_rmse'].mean():.2e}\n")
                f.write(f"- Std RMSE: {subset['primary_rmse'].std():.2e}\n")
                f.write(f"- Mean R²: {subset['primary_r2'].mean():.4f}\n")
                f.write(f"- Std R²: {subset['primary_r2'].std():.4f}\n")
                f.write(f"- Mean simulations: {subset['total_simulations'].mean():.1f}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| Approach | Model | RMSE | R² | Total Sims | Simulator | Chemistry |\n")
            f.write("|----------|-------|------|----|-----------|-----------|-----------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['approach']} | {row['model_type']} | {row['primary_rmse']:.2e} | "
                       f"{row['primary_r2']:.4f} | {row['total_simulations']} | "
                       f"{row['simulator']} | {row['chemistry']} |\n")
            
            f.write("\n## Best Performers\n\n")
            
            # Best by RMSE
            best_rmse = df.loc[df['primary_rmse'].idxmin()]
            f.write(f"**Lowest RMSE:** {best_rmse['approach']} approach with {best_rmse['model_type']} "
                   f"(RMSE: {best_rmse['primary_rmse']:.2e}, R²: {best_rmse['primary_r2']:.4f})\n\n")
            
            # Best by R²
            best_r2 = df.loc[df['primary_r2'].idxmax()]
            f.write(f"**Highest R²:** {best_r2['approach']} approach with {best_r2['model_type']} "
                   f"(R²: {best_r2['primary_r2']:.4f}, RMSE: {best_r2['primary_rmse']:.2e})\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `comparison_summary.csv` - Combined results data\n")
            f.write("- `rmse_comparison.png` - RMSE comparison plots\n")
            f.write("- `r2_comparison.png` - R² score comparison plots\n")
            f.write("- `efficiency_comparison.png` - Efficiency analysis\n")
            f.write("- `model_specific_comparison.png` - Model-specific comparisons\n")
        
        print(f"   Report saved: {report_file}")
    
    def save_combined_results(self, df):
        """Save the combined results CSV."""
        output_file = self.comparison_dir / 'comparison_summary.csv'
        df.to_csv(output_file, index=False)
        print(f"   Combined results saved: {output_file}")
    
    def run_comparison(self):
        """Run the complete comparison analysis."""
        print("=" * 80)
        print("APPROACH COMPARISON ANALYSIS")
        print("=" * 80)
        
        # Find result files
        traditional_files, adaptive_files = self.find_result_files()
        
        if not traditional_files and not adaptive_files:
            print("❌ No result files found! Please run the individual analyses first.")
            return
        
        # Load and combine results
        print("\n📂 Loading results...")
        combined_df = self.load_and_combine_results(traditional_files, adaptive_files)
        
        # Standardize metrics
        print("\n🔧 Standardizing metrics...")
        combined_df = self.standardize_metrics(combined_df)
        
        # Generate plots
        self.generate_comparison_plots(combined_df)
        
        # Generate report
        self.generate_summary_report(combined_df)
        
        # Save combined results
        self.save_combined_results(combined_df)
        
        print(f"\n✅ Comparison analysis complete! Results saved to {self.comparison_dir}")


def main():
    """Main comparison runner."""
    comparator = ApproachComparison()
    comparator.run_comparison()


if __name__ == "__main__":
    main()
