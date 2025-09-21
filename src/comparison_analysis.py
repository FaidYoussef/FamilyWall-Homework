from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
import json
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DomainScores(BaseModel):
    """Scores for a single domain evaluation"""
    length: int 
    extension: int = Field(ge=0, le=5)
    relevance: int = Field(ge=0, le=5)
    brandability: int = Field(ge=0, le=5)
    literalness: int = Field(ge=0, le=5)
    style: int = Field(ge=0, le=5)
    safety: int = Field(ge=0, le=5)
    
    def get_total_score(self) -> int:
        """Calculate total score across all criteria"""
        return sum([self.length, self.extension, self.relevance, 
                   self.brandability, self.literalness, self.style, self.safety])
    
    def get_weighted_score(self) -> float:
        """Calculate weighted score prioritizing key business metrics"""
        weights = {
            'length': 1.0,
            'extension': 0.8,
            'relevance': 1.5,
            'brandability': 1.5,
            'literalness': 1.2,
            'style': 1.0,
            'safety': 2.0  # Safety is most important
        }
        
        weighted_sum = (
            self.length * weights['length'] +
            self.extension * weights['extension'] +
            self.relevance * weights['relevance'] +
            self.brandability * weights['brandability'] +
            self.literalness * weights['literalness'] +
            self.style * weights['style'] +
            self.safety * weights['safety']
        )
        
        total_weights = sum(weights.values())
        return weighted_sum / total_weights

class DomainEvaluation(BaseModel):
    """Individual domain evaluation"""
    domain: str
    scores: DomainScores
    issues: List[str] = Field(default_factory=list)
    comments: str = ""
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if domain has critical issues"""
        critical_issues = {'unsafe', 'wrong_extension', 'irrelevant'}
        return bool(set(self.issues) & critical_issues)
    
    @property
    def is_acceptable(self) -> bool:
        """Determine if domain meets minimum quality standards"""
        return (
            not self.has_critical_issues and
            self.scores.safety >= 4 and
            self.scores.relevance >= 3 and
            self.scores.extension >= 4
        )

class BusinessEvaluation(BaseModel):
    """Evaluation for all domains of a single business"""
    business_description: str
    model_name: str
    evaluation: Dict[str, List[DomainEvaluation]]
    
    def get_best_domain(self) -> Optional[DomainEvaluation]:
        """Get the highest scoring acceptable domain"""
        evaluations = self.evaluation.get('evaluations', [])
        acceptable_domains = [d for d in evaluations if d.is_acceptable]
        
        if not acceptable_domains:
            return None
            
        return max(acceptable_domains, key=lambda d: d.scores.get_weighted_score())
    
    def get_success_rate(self) -> float:
        """Calculate percentage of acceptable domains"""
        evaluations = self.evaluation.get('evaluations', [])
        if not evaluations:
            return 0.0
        
        acceptable_count = sum(1 for d in evaluations if d.is_acceptable)
        return acceptable_count / len(evaluations)

@dataclass
class ModelComparisonStats:
    """Statistical comparison between two models"""
    model_a_name: str
    model_b_name: str
    total_businesses: int
    
    # Success rates
    model_a_success_rate: float
    model_b_success_rate: float
    success_rate_improvement: float
    
    # Average scores
    model_a_avg_score: float
    model_b_avg_score: float
    avg_score_improvement: float
    
    # Statistical significance
    score_p_value: float
    success_rate_p_value: float
    is_significantly_better: bool
    
    # Issue analysis
    common_issues: Dict[str, Tuple[int, int]]  # issue -> (model_a_count, model_b_count)
    
    def print_summary(self):
        """Print a comprehensive comparison summary"""
        print(f"=== Model Comparison: {self.model_a_name} vs {self.model_b_name} ===")
        print(f"Total businesses analyzed: {self.total_businesses}")
        print(f"\n📊 SUCCESS RATES:")
        print(f"  {self.model_a_name}: {self.model_a_success_rate:.1%}")
        print(f"  {self.model_b_name}: {self.model_b_success_rate:.1%}")
        print(f"  Improvement: {self.success_rate_improvement:+.1%}")
        
        print(f"\n⭐ AVERAGE QUALITY SCORES:")
        print(f"  {self.model_a_name}: {self.model_a_avg_score:.2f}/5.0")
        print(f"  {self.model_b_name}: {self.model_b_avg_score:.2f}/5.0")
        print(f"  Improvement: {self.avg_score_improvement:+.2f}")
        
        print(f"\n🔬 STATISTICAL SIGNIFICANCE:")
        print(f"  Score difference p-value: {self.score_p_value:.4f}")
        print(f"  Success rate difference p-value: {self.success_rate_p_value:.4f}")
        print(f"  Significantly better: {'Yes' if self.is_significantly_better else 'No'}")
        
        print(f"\n🚨 COMMON ISSUES COMPARISON:")
        for issue, (count_a, count_b) in sorted(self.common_issues.items(), 
                                               key=lambda x: x[1][1], reverse=True):
            if count_a > 0 or count_b > 0:
                print(f"  {issue}: {self.model_a_name}={count_a}, {self.model_b_name}={count_b}")

class DomainModelAnalyzer:
    """Main analyzer class for comparing domain generation models"""
    
    def __init__(self):
        self.evaluations_data: Dict[str, List[BusinessEvaluation]] = {}
        
    def load_evaluations(self, model_name: str, data: List[Dict[str, Any]]) -> None:
        """Load evaluation data for a model"""
        evaluations = []
        
        for item in data:
            # Parse domain evaluations
            eval_data = item.get('evaluation', {})
            domain_evals = []
            
            for eval_item in eval_data.get('evaluations', []):
                domain_eval = DomainEvaluation(
                    domain=eval_item['domain'],
                    scores=DomainScores(**eval_item['scores']),
                    issues=eval_item.get('issues', []),
                    comments=eval_item.get('comments', '')
                )
                domain_evals.append(domain_eval)
            
            business_eval = BusinessEvaluation(
                business_description=item.get('business_description', ''),
                model_name=model_name,
                evaluation={'evaluations': domain_evals}
            )
            evaluations.append(business_eval)
        
        self.evaluations_data[model_name] = evaluations
        print(f"✅ Loaded {len(evaluations)} business evaluations for {model_name}")
    
    def compare_models(self, model_a: str, model_b: str) -> ModelComparisonStats:
        """Compare two models and return detailed statistics"""
        if model_a not in self.evaluations_data or model_b not in self.evaluations_data:
            raise ValueError(f"Models {model_a} and/or {model_b} not found in loaded data")
        
        evals_a = self.evaluations_data[model_a]
        evals_b = self.evaluations_data[model_b]
        
        # Ensure same number of businesses
        min_count = min(len(evals_a), len(evals_b))
        evals_a = evals_a[:min_count]
        evals_b = evals_b[:min_count]
        
        # Calculate success rates
        success_rates_a = [eval.get_success_rate() for eval in evals_a]
        success_rates_b = [eval.get_success_rate() for eval in evals_b]
        
        avg_success_a = np.mean(success_rates_a)
        avg_success_b = np.mean(success_rates_b)
        
        # Calculate average quality scores
        scores_a = []
        scores_b = []
        
        for eval in evals_a:
            for domain_eval in eval.evaluation['evaluations']:
                scores_a.append(domain_eval.scores.get_weighted_score())
        
        for eval in evals_b:
            for domain_eval in eval.evaluation['evaluations']:
                scores_b.append(domain_eval.scores.get_weighted_score())
        
        avg_score_a = np.mean(scores_a) if scores_a else 0
        avg_score_b = np.mean(scores_b) if scores_b else 0
        
        # Statistical significance tests
        score_p_value = stats.ttest_ind(scores_a, scores_b)[1] if scores_a and scores_b else 1.0
        # print("scores_a, scores_b = ")
        # print(scores_a, scores_b)
        success_p_value = stats.ttest_ind(success_rates_a, success_rates_b)[1]
        # print("success_rates_a, success_rates_b = ")
        # print(success_rates_a, success_rates_b)
        
        # Issue analysis
        issues_a = Counter()
        issues_b = Counter()
        
        for eval in evals_a:
            for domain_eval in eval.evaluation['evaluations']:
                issues_a.update(domain_eval.issues)
        
        for eval in evals_b:
            for domain_eval in eval.evaluation['evaluations']:
                issues_b.update(domain_eval.issues)
        
        all_issues = set(issues_a.keys()) | set(issues_b.keys())
        common_issues = {issue: (issues_a[issue], issues_b[issue]) for issue in all_issues}
        
        return ModelComparisonStats(
            model_a_name=model_a,
            model_b_name=model_b,
            total_businesses=min_count,
            model_a_success_rate=avg_success_a,
            model_b_success_rate=avg_success_b,
            success_rate_improvement=avg_success_b - avg_success_a,
            model_a_avg_score=avg_score_a,
            model_b_avg_score=avg_score_b,
            avg_score_improvement=avg_score_b - avg_score_a,
            score_p_value=score_p_value,
            success_rate_p_value=success_p_value,
            is_significantly_better=(score_p_value < 0.05 and avg_score_b > avg_score_a),
            common_issues=common_issues
        )
    
    def analyze_model_strengths_weaknesses(self, model_name: str) -> Dict[str, Any]:
        """Detailed analysis of a single model's performance"""
        if model_name not in self.evaluations_data:
            raise ValueError(f"Model {model_name} not found")
        
        evaluations = self.evaluations_data[model_name]
        
        # Aggregate all domain evaluations
        all_domains = []
        for business_eval in evaluations:
            all_domains.extend(business_eval.evaluation['evaluations'])
        
        # Score distribution analysis
        score_breakdown = {
            'length': [d.scores.length for d in all_domains],
            'extension': [d.scores.extension for d in all_domains],
            'relevance': [d.scores.relevance for d in all_domains],
            'brandability': [d.scores.brandability for d in all_domains],
            'literalness': [d.scores.literalness for d in all_domains],
            'style': [d.scores.style for d in all_domains],
            'safety': [d.scores.safety for d in all_domains]
        }
        
        # Calculate statistics for each criterion
        criterion_stats = {}
        for criterion, scores in score_breakdown.items():
            criterion_stats[criterion] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': min(scores),
                'max': max(scores),
                'below_threshold': sum(1 for s in scores if s < 3) / len(scores)
            }
        
        # Issue frequency analysis
        all_issues = []
        for domain in all_domains:
            all_issues.extend(domain.issues)
        
        issue_counts = Counter(all_issues)
        
        # Quality tier analysis
        quality_tiers = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}
        
        for domain in all_domains:
            weighted_score = domain.scores.get_weighted_score()
            if weighted_score >= 4.5:
                quality_tiers['excellent'] += 1
            elif weighted_score >= 4.0:
                quality_tiers['good'] += 1
            elif weighted_score >= 3.0:
                quality_tiers['acceptable'] += 1
            else:
                quality_tiers['poor'] += 1
        
        return {
            'model_name': model_name,
            'total_domains': len(all_domains),
            'total_businesses': len(evaluations),
            'criterion_performance': criterion_stats,
            'issue_frequency': dict(issue_counts),
            'quality_distribution': quality_tiers,
            'overall_success_rate': sum(1 for d in all_domains if d.is_acceptable) / len(all_domains),
            'avg_weighted_score': np.mean([d.scores.get_weighted_score() for d in all_domains])
        }
    
    def generate_improvement_recommendations(self, model_name: str) -> List[str]:
        """Generate specific recommendations for model improvement"""
        analysis = self.analyze_model_strengths_weaknesses(model_name)
        recommendations = []
        
        # Check criterion performance
        criterion_perf = analysis['criterion_performance']
        
        if criterion_perf['relevance']['mean'] < 4.0:
            recommendations.append(
                "🎯 RELEVANCE: Add more business-specific keywords and context understanding to training data"
            )
        
        if criterion_perf['brandability']['mean'] < 4.0:
            recommendations.append(
                "🏷️ BRANDABILITY: Include more creative, memorable domain examples and reduce generic suggestions"
            )
        
        if criterion_perf['literalness']['mean'] < 4.0:
            recommendations.append(
                "✂️ LITERALNESS: Train model to create shorter, more abstract versions of business names"
            )
        
        if criterion_perf['length']['mean'] < 4.0:
            recommendations.append(
                "📏 LENGTH: Add constraints to prefer shorter domains (6-12 characters)"
            )
        
        # Check issue frequency
        issue_freq = analysis['issue_frequency']
        total_domains = analysis['total_domains']
        
        if issue_freq.get('too_literal', 0) / total_domains > 0.2:
            recommendations.append(
                "🔄 Implement post-processing to detect and replace overly literal suggestions"
            )
        
        if issue_freq.get('too_long', 0) / total_domains > 0.15:
            recommendations.append(
                "⚡ Add length penalty in training objective or sampling strategy"
            )
        
        if issue_freq.get('not_brandable', 0) / total_domains > 0.1:
            recommendations.append(
                "💡 Augment training data with successful brand examples and naming patterns"
            )
        
        # Overall performance recommendations
        if analysis['overall_success_rate'] < 0.6:
            recommendations.append(
                "📊 Overall success rate is low - consider ensemble methods or multi-step generation"
            )
        
        if not recommendations:
            recommendations.append("🎉 Model performance is strong across all criteria!")
        
        return recommendations

# Example usage and testing functions
def load_sample_data() -> Tuple[List[Dict], List[Dict]]:
    """Create sample data for testing (replace with your actual data loading)"""
    
    sample_mistral = [{
        "business_description": "Spanish tapas restaurant in Flatiron District NYC specializing in authentic Iberian cuisine",
        "evaluation": {
            "evaluations": [
                {
                    "domain": "FlatironTapas.com",
                    "scores": {
                        "length": 5, "extension": 5, "relevance": 4,
                        "brandability": 4, "literalness": 2, "style": 5, "safety": 5
                    },
                    "issues": ["too_literal"],
                    "comments": "Too literal copy of business description"
                },
                {
                    "domain": "BoqueriaNYC.com", 
                    "scores": {
                        "length": 4, "extension": 5, "relevance": 5,
                        "brandability": 5, "literalness": 4, "style": 5, "safety": 5
                    },
                    "issues": ["acceptable"],
                    "comments": "Good balance of relevance and brandability"
                }
            ]
        }
    }]
    
    sample_gemma = [{
        "business_description": "Spanish tapas restaurant in Flatiron District NYC specializing in authentic Iberian cuisine",
        "evaluation": {
            "evaluations": [
                {
                    "domain": "TapasFlatiron.com",
                    "scores": {
                        "length": 5, "extension": 5, "relevance": 5,
                        "brandability": 4, "literalness": 3, "style": 5, "safety": 5
                    },
                    "issues": ["acceptable"],
                    "comments": "Well balanced domain suggestion"
                },
                {
                    "domain": "IberianBites.com",
                    "scores": {
                        "length": 4, "extension": 5, "relevance": 4,
                        "brandability": 5, "literalness": 5, "style": 5, "safety": 5
                    },
                    "issues": ["acceptable"],
                    "comments": "Creative and brandable"
                }
            ]
        }
    }]
    
    return sample_mistral, sample_gemma

def run_example_analysis():
    """Example of how to use the analyzer"""
    
    # Initialize analyzer
    analyzer = DomainModelAnalyzer()
    
    # Load sample data (replace with your actual data)
    mistral_data, gemma_data = load_sample_data()
    
    analyzer.load_evaluations('mistral', mistral_data)
    analyzer.load_evaluations('gemma', gemma_data)
    
    # Compare models
    print("\n" + "="*60)
    comparison = analyzer.compare_models('mistral', 'gemma')
    comparison.print_summary()
    
    # Individual model analysis
    print(f"\n{'='*60}")
    print("MISTRAL DETAILED ANALYSIS:")
    mistral_analysis = analyzer.analyze_model_strengths_weaknesses('mistral')
    print(f"Success Rate: {mistral_analysis['overall_success_rate']:.1%}")
    print(f"Avg Quality Score: {mistral_analysis['avg_weighted_score']:.2f}/5.0")
    
    print(f"\nGEMMA DETAILED ANALYSIS:")
    gemma_analysis = analyzer.analyze_model_strengths_weaknesses('gemma')
    print(f"Success Rate: {gemma_analysis['overall_success_rate']:.1%}")
    print(f"Avg Quality Score: {gemma_analysis['avg_weighted_score']:.2f}/5.0")
    
    # Improvement recommendations
    print(f"\n{'='*60}")
    print("IMPROVEMENT RECOMMENDATIONS FOR MISTRAL:")
    mistral_recommendations = analyzer.generate_improvement_recommendations('mistral')
    for rec in mistral_recommendations:
        print(f"  • {rec}")
    
    return analyzer, comparison


def load_data_from_json(mistral_path: str, gemma_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load domain evaluation data for mistral and gemma models from JSON files.
    
    Args:
        mistral_path (str): Path to the JSON file containing mistral data.
        gemma_path (str): Path to the JSON file containing gemma data.
    
    Returns:
        Tuple[List[Dict], List[Dict]]: Loaded mistral and gemma data.
    """
    with open(mistral_path, "r", encoding="utf-8") as f:
        mistral_data = json.load(f)

    with open(gemma_path, "r", encoding="utf-8") as f:
        gemma_data = json.load(f)

    return mistral_data, gemma_data


def run_analysis(mistral_file: str, gemma_file: str):
    analyzer = DomainModelAnalyzer()

    # Load from JSON instead of sample data
    mistral_data, gemma_data = load_data_from_json(mistral_file, gemma_file)
    
    analyzer.load_evaluations('mistral', mistral_data)
    analyzer.load_evaluations('gemma', gemma_data)

    # The rest stays the same
    print("\n" + "="*60)
    comparison = analyzer.compare_models('mistral', 'gemma')
    comparison.print_summary()
    
    return analyzer, comparison
