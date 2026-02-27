"""
AB实验分析模块：实验效果评估、统计检验
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABTestAnalyzer:
    """AB实验分析器"""
    
    def __init__(self, alpha: float = 0.05, min_effect: float = 0.05):
        """
        初始化
        
        Args:
            alpha: 显著性水平
            min_effect: 最小可检测效应
        """
        self.alpha = alpha
        self.min_effect = min_effect
        self.results = {}
        
    def load_experiment_data(self, 
                            queries_df: pd.DataFrame,
                            feedback_df: pd.DataFrame,
                            responses_df: pd.DataFrame,
                            experiment_id: str) -> pd.DataFrame:
        """
        加载实验数据
        
        Args:
            queries_df: 查询表
            feedback_df: 反馈表
            responses_df: 回答表
            experiment_id: 实验ID
            
        Returns:
            合并后的实验数据
        """
        logger.info(f"Loading data for experiment {experiment_id}")
        
        # 筛选实验组数据
        exp_queries = queries_df[queries_df['experiment_group'].notna()].copy()
        
        # 合并反馈数据
        exp_data = exp_queries.merge(
            feedback_df[['query_id', 'rating', 'nps_score']],
            on='query_id',
            how='left'
        )
        
        # 合并回答数据
        exp_data = exp_data.merge(
            responses_df[['query_id', 'retrieval_method', 'response_time_ms']],
            on='query_id',
            how='left'
        )
        
        # 创建满意度标签
        exp_data['is_satisfied'] = (exp_data['rating'] >= 4).astype(int)
        
        logger.info(f"Loaded {len(exp_data)} records for {exp_data['experiment_group'].nunique()} groups")
        return exp_data
    
    def calculate_metrics(self, df: pd.DataFrame, group_col: str = 'experiment_group') -> pd.DataFrame:
        """
        计算各实验组的核心指标
        
        Args:
            df: 实验数据
            group_col: 分组列名
            
        Returns:
            各组的指标统计
        """
        metrics = []
        
        for group, group_df in df.groupby(group_col):
            metric = {
                'group': group,
                'sample_size': len(group_df),
                'satisfaction_rate': group_df['is_satisfied'].mean() * 100,
                'avg_rating': group_df['rating'].mean(),
                'avg_nps': group_df['nps_score'].mean(),
                'avg_response_time': group_df['response_time_ms'].mean(),
                'rating_std': group_df['rating'].std(),
                'nps_std': group_df['nps_score'].std()
            }
            metrics.append(metric)
        
        metrics_df = pd.DataFrame(metrics)
        logger.info(f"Calculated metrics for {len(metrics)} groups")
        
        return metrics_df
    
    def conduct_z_test(self, 
                       control_df: pd.DataFrame,
                       treatment_df: pd.DataFrame,
                       metric: str = 'is_satisfied') -> Dict:
        """
        进行Z检验（比例检验）
        
        Args:
            control_df: 对照组数据
            treatment_df: 实验组数据
            metric: 检验的指标
            
        Returns:
            检验结果
        """
        control_success = control_df[metric].sum()
        control_n = len(control_df)
        treatment_success = treatment_df[metric].sum()
        treatment_n = len(treatment_df)
        
        # 比例Z检验
        z_stat, p_value = proportions_ztest(
            count=[treatment_success, control_success],
            nobs=[treatment_n, control_n]
        )
        
        # 计算置信区间
        control_rate = control_success / control_n
        treatment_rate = treatment_success / treatment_n
        diff = treatment_rate - control_rate
        se = np.sqrt(
            control_rate * (1 - control_rate) / control_n + 
            treatment_rate * (1 - treatment_rate) / treatment_n
        )
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se
        
        # 相对提升
        relative_lift = diff / control_rate if control_rate > 0 else 0
        
        result = {
            'metric': metric,
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'absolute_lift': diff,
            'relative_lift': relative_lift,
            'z_stat': z_stat,
            'p_value': p_value,
            'ci_95': (ci_lower, ci_upper),
            'is_significant': p_value < self.alpha
        }
        
        return result
    
    def conduct_ttest(self,
                     control_df: pd.DataFrame,
                     treatment_df: pd.DataFrame,
                     metric: str = 'rating') -> Dict:
        """
        进行T检验（均值检验）
        
        Args:
            control_df: 对照组数据
            treatment_df: 实验组数据
            metric: 检验的指标
            
        Returns:
            检验结果
        """
        control_values = control_df[metric].dropna()
        treatment_values = treatment_df[metric].dropna()
        
        # 独立样本T检验
        t_stat, p_value = stats.ttest_ind(
            treatment_values, 
            control_values,
            equal_var=False  # Welch's t-test
        )
        
        # 计算效应量 Cohen's d
        control_mean = control_values.mean()
        treatment_mean = treatment_values.mean()
        control_std = control_values.std()
        treatment_std = treatment_values.std()
        
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        result = {
            'metric': metric,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'difference': treatment_mean - control_mean,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': p_value < self.alpha
        }
        
        return result
    
    def run_analysis(self, 
                    exp_data: pd.DataFrame,
                    control_group: str = 'control') -> Dict:
        """
        运行完整的AB实验分析
        
        Args:
            exp_data: 实验数据
            control_group: 对照组名称
            
        Returns:
            完整的分析结果
        """
        logger.info(f"Running AB test analysis with control group: {control_group}")
        
        # 分离对照组
        control_df = exp_data[exp_data['experiment_group'] == control_group]
        
        if control_df.empty:
            logger.error(f"Control group {control_group} not found")
            return {}
        
        # 计算整体指标
        metrics_df = self.calculate_metrics(exp_data)
        
        # 对每个实验组进行检验
        treatment_groups = [g for g in exp_data['experiment_group'].unique() if g != control_group]
        
        results = {
            'overall_metrics': metrics_df.to_dict('records'),
            'treatment_results': {}
        }
        
        for group in treatment_groups:
            treatment_df = exp_data[exp_data['experiment_group'] == group]
            
            group_results = {}
            
            # 对比例指标进行Z检验
            group_results['satisfaction'] = self.conduct_z_test(
                control_df, treatment_df, 'is_satisfied'
            )
            
            # 对连续指标进行T检验
            group_results['rating'] = self.conduct_ttest(
                control_df, treatment_df, 'rating'
            )
            
            group_results['nps'] = self.conduct_ttest(
                control_df, treatment_df, 'nps_score'
            )
            
            # 幂分析（检查样本量是否足够）
            effect_size = group_results['satisfaction']['absolute_lift']
            power_analysis = self.calculate_power(
                effect_size=effect_size,
                sample_size=len(treatment_df),
                control_size=len(control_df)
            )
            group_results['power_analysis'] = power_analysis
            
            results['treatment_results'][group] = group_results
            
            logger.info(f"Group {group}: satisfaction lift = {group_results['satisfaction']['absolute_lift']:.2%}, "
                       f"p_value = {group_results['satisfaction']['p_value']:.4f}")
        
        self.results = results
        return results
    
    def calculate_power(self, 
                        effect_size: float,
                        sample_size: int,
                        control_size: int,
                        alpha: float = None) -> Dict:
        """
        计算统计功效
        
        Args:
            effect_size: 效应量
            sample_size: 实验组样本量
            control_size: 对照组样本量
            alpha: 显著性水平
            
        Returns:
            功效分析结果
        """
        if alpha is None:
            alpha = self.alpha
        
        # 使用statsmodels计算功效
        power_analysis = TTestIndPower()
        
        # 计算效应量（Cohen's h for proportions）
        from statsmodels.stats.proportion import proportion_effectsize
        
        # 假设对照组比例为0.5，计算Cohen's h
        cohens_h = proportion_effectsize(0.5, 0.5 + effect_size)
        
        # 计算功效
        power = power_analysis.power(
            effect_size=cohens_h,
            nobs1=sample_size,
            alpha=alpha,
            ratio=control_size/sample_size,
            alternative='two-sided'
        )
        
        # 计算所需样本量（给定80%功效）
        required_n = power_analysis.solve_power(
            effect_size=cohens_h,
            power=0.8,
            alpha=alpha,
            ratio=control_size/sample_size,
            alternative='two-sided'
        )
        
        return {
            'power': power,
            'is_adequate': power >= 0.8,
            'required_sample_size': int(required_n),
            'current_sample_size': sample_size
        }
    
    def visualize_results(self, results: Dict = None, save_path: str = None):
        """
        可视化实验结果
        
        Args:
            results: 分析结果
            save_path: 保存路径
        """
        if results is None:
            results = self.results
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 各组满意度对比
        ax1 = axes[0, 0]
        metrics_df = pd.DataFrame(results['overall_metrics'])
        groups = metrics_df['group']
        satisfaction = metrics_df['satisfaction_rate']
        
        colors = ['red' if g == 'control' else 'blue' for g in groups]
        bars = ax1.bar(groups, satisfaction, color=colors)
        ax1.set_title('Satisfaction Rate by Group', fontsize=14)
        ax1.set_ylabel('Satisfaction Rate (%)')
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, val in zip(bars, satisfaction):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontsize=10)
        
        # 2. 各组NPS对比
        ax2 = axes[0, 1]
        nps = metrics_df['avg_nps']
        bars = ax2.bar(groups, nps, color=colors)
        ax2.set_title('Average NPS by Group', fontsize=14)
        ax2.set_ylabel('NPS Score')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, nps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=10)
        
        # 3. 置信区间图
        ax3 = axes[1, 0]
        
        treatment_data = []
        for group, group_results in results['treatment_results'].items():
            sat_result = group_results['satisfaction']
            treatment_data.append({
                'group': group,
                'lift': sat_result['absolute_lift'] * 100,
                'ci_lower': sat_result['ci_95'][0] * 100,
                'ci_upper': sat_result['ci_95'][1] * 100,
                'significant': sat_result['is_significant']
            })
        
        treatment_df = pd.DataFrame(treatment_data)
        
        for idx, row in treatment_df.iterrows():
            color = 'green' if row['significant'] else 'gray'
            ax3.errorbar(x=row['lift'], y=row['group'], 
                        xerr=[[row['lift'] - row['ci_lower']], [row['ci_upper'] - row['lift']]],
                        fmt='o', color=color, capsize=5, capthick=2)
        
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('Treatment Effect with 95% CI', fontsize=14)
        ax3.set_xlabel('Absolute Lift in Satisfaction (%)')
        
        # 4. 功效分析
        ax4 = axes[1, 1]
        
        power_data = []
        for group, group_results in results['treatment_results'].items():
            power_data.append({
                'group': group,
                'power': group_results['power_analysis']['power']
            })
        
        power_df = pd.DataFrame(power_data)
        bars = ax4.bar(power_df['group'], power_df['power'])
        ax4.axhline(y=0.8, color='red', linestyle='--', label='80% threshold')
        ax4.set_title('Statistical Power by Group', fontsize=14)
        ax4.set_ylabel('Power')
        ax4.set_ylim(0, 1)
        ax4.legend()
        
        for bar, val in zip(bars, power_df['power']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, results: Dict = None, format: str = 'markdown') -> str:
        """
        生成实验报告
        
        Args:
            results: 分析结果
            format: 报告格式
            
        Returns:
            报告文本
        """
        if results is None:
            results = self.results
        
        if format == 'markdown':
            report = "# AB实验分析报告\n\n"
            
            # 整体指标
            report += "## 1. 整体指标\n\n"
            report += "| 实验组 | 样本量 | 满意度 | 平均评分 | 平均NPS | 响应时间(ms) |\n"
            report += "|--------|--------|--------|----------|---------|--------------|\n"
            
            for metric in results['overall_metrics']:
                report += f"| {metric['group']} | {metric['sample_size']} | "
                report += f"{metric['satisfaction_rate']:.1f}% | {metric['avg_rating']:.2f} | "
                report += f"{metric['avg_nps']:.