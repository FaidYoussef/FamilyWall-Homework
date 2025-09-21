import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import Counter
import warnings
from comparison_analysis import DomainModelAnalyzer, load_data_from_json
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class DomainAnalysisVisualizer:
    """Visualization tools for domain generation model analysis"""
    
    def __init__(self, analyzer):
        """Initialize with a DomainModelAnalyzer instance"""
        self.analyzer = analyzer
        
    def plot_model_comparison_overview(self, model_a: str, model_b: str,
                                   save_prefix: str = None) -> None:
        """Generate and save/show multiple figures comparing two models."""

        comparison = self.analyzer.compare_models(model_a, model_b)
        analysis_a = self.analyzer.analyze_model_strengths_weaknesses(model_a)
        analysis_b = self.analyzer.analyze_model_strengths_weaknesses(model_b)

        # 1. Success Rate Comparison
        plt.figure(figsize=(6, 4))
        success_rates = [comparison.model_a_success_rate, comparison.model_b_success_rate]
        colors = ['lightcoral', 'lightblue']
        bars = plt.bar([model_a, model_b], success_rates, color=colors, alpha=0.7)
        plt.title('Success Rate Comparison', fontweight='bold')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        if save_prefix:
            plt.savefig(f"{save_prefix}_success_rate.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        # 2. Average Quality Score
        plt.figure(figsize=(6, 4))
        avg_scores = [comparison.model_a_avg_score, comparison.model_b_avg_score]
        bars = plt.bar([model_a, model_b], avg_scores, color=colors, alpha=0.7)
        plt.title('Average Quality Score', fontweight='bold')
        plt.ylabel('Weighted Score')
        plt.ylim(0, 5)
        for bar, score in zip(bars, avg_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        if save_prefix:
            plt.savefig(f"{save_prefix}_avg_quality.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        # 3. Quality Distribution
        plt.figure(figsize=(7, 5))
        quality_a = analysis_a['quality_distribution']
        quality_b = analysis_b['quality_distribution']
        categories = list(quality_a.keys())
        values_a = [quality_a[c] for c in categories]
        values_b = [quality_b[c] for c in categories]
        x = np.arange(len(categories))
        width = 0.35
        plt.bar(x - width/2, values_a, width, label=model_a, alpha=0.7, color='lightcoral')
        plt.bar(x + width/2, values_b, width, label=model_b, alpha=0.7, color='lightblue')
        plt.title('Quality Distribution', fontweight='bold')
        plt.ylabel('Number of Domains')
        plt.xticks(x, categories, rotation=45)
        plt.legend()
        if save_prefix:
            plt.savefig(f"{save_prefix}_quality_dist.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        # 4. Criterion Performance Radar
        criteria = ['length', 'extension', 'relevance', 'brandability', 'literalness', 'style', 'safety']
        perf_a = analysis_a['criterion_performance']
        perf_b = analysis_b['criterion_performance']
        values_a = [perf_a[c]['mean'] for c in criteria]
        values_b = [perf_b[c]['mean'] for c in criteria]
        values_a += values_a[:1]
        values_b += values_b[:1]
        criteria_plot = criteria + [criteria[0]]
        angles = np.linspace(0, 2 * np.pi, len(criteria_plot))
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values_a, 'o-', linewidth=2, label=model_a, color='red')
        ax.fill(angles, values_a, alpha=0.25, color='red')
        ax.plot(angles, values_b, 'o-', linewidth=2, label=model_b, color='blue')
        ax.fill(angles, values_b, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria, fontsize=9)
        ax.set_ylim(0, 5)
        ax.set_title('Criterion Performance', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        if save_prefix:
            plt.savefig(f"{save_prefix}_criterion_radar.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        # 5. Issue Frequency
        plt.figure(figsize=(12, 6))
        issues = comparison.common_issues
        if issues:
            issue_names = list(issues.keys())
            counts_a = [issues[i][0] for i in issue_names]
            counts_b = [issues[i][1] for i in issue_names]
            x = np.arange(len(issue_names))
            width = 0.35
            plt.bar(x - width/2, counts_a, width, label=model_a, alpha=0.7, color='lightcoral')
            plt.bar(x + width/2, counts_b, width, label=model_b, alpha=0.7, color='lightblue')
            plt.title('Issue Frequency', fontweight='bold')
            plt.ylabel('Count')
            plt.xticks(x, issue_names, rotation=45, ha='right')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No significant issues found', ha='center', va='center')
            plt.title('Issue Frequency', fontweight='bold')
        if save_prefix:
            plt.savefig(f"{save_prefix}_issues.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        # 6. Statistical Summary (text figure)
        plt.figure(figsize=(6, 4))
        plt.axis('off')
        summary_text = f"""
    Statistical Summary

    Success Rate Improvement: {comparison.success_rate_improvement:+.1%}
    p-value: {comparison.success_rate_p_value:.4f}

    Quality Score Improvement: {comparison.avg_score_improvement:+.2f}
    p-value: {comparison.score_p_value:.4f}

    Significantly Better: {'✅ YES' if comparison.is_significantly_better else '❌ NO'}
    Total Businesses: {comparison.total_businesses}
    """
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.title("Statistical Summary", fontweight="bold")
        if save_prefix:
            plt.savefig(f"{save_prefix}_stats.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

    
    def plot_criterion_heatmap(self, models: List[str], 
                               figsize: tuple = (12, 8),
                               save_path: str = None) -> None:
        """Create heatmap showing performance across all criteria for multiple models"""
        
        criteria = ['length', 'extension', 'relevance', 'brandability', 'literalness', 'style', 'safety']
        heatmap_data = []
        
        for model in models:
            analysis = self.analyzer.analyze_model_strengths_weaknesses(model)
            criterion_perf = analysis['criterion_performance']
            model_scores = [criterion_perf[criterion]['mean'] for criterion in criteria]
            heatmap_data.append(model_scores)
        
        df_heatmap = pd.DataFrame(heatmap_data, index=models, columns=criteria)
        
        plt.figure(figsize=figsize)
        sns.heatmap(df_heatmap, annot=True, cmap="coolwarm", cbar_kws={'label': 'Average Score'}, 
                    linewidths=0.5, linecolor='gray', fmt=".2f")
        
        plt.title("Criterion Performance Heatmap", fontsize=16, fontweight="bold", pad=20)
        plt.ylabel("Models", fontsize=12, fontweight="bold")
        plt.xlabel("Criteria", fontsize=12, fontweight="bold")
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved figure to {save_path}")
        else:
            plt.show()


def run_visualization(mistral_file: str, gemma_file: str):
    analyzer = DomainModelAnalyzer()
    
    # Load from JSON files
    mistral_data, gemma_data = load_data_from_json(mistral_file, gemma_file)
    analyzer.load_evaluations('mistral', mistral_data)
    analyzer.load_evaluations('gemma', gemma_data)
    
    # Initialize visualizer
    visualizer = DomainAnalysisVisualizer(analyzer)
    
    # Save comparison overview
    visualizer.plot_model_comparison_overview(
        'mistral', 'gemma',
        save_prefix="../figures/comparison_overview"
    )
    
    # Save heatmap
    visualizer.plot_criterion_heatmap(
        ['mistral', 'gemma'],
        save_path="../figures/criterion_heatmap.png"
    )
