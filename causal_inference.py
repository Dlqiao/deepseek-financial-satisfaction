"""
因果推断模块：PSM、DID等因果分析方法实现
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropensityScoreMatcher:
    """
    倾向性评分匹配(PSM)实现
    """
    
    def __init__(self, caliper: float = 0.2, ratio: int = 1, replace: bool = False):
        """
        初始化PSM匹配器
        
        Args:
            caliper: 匹配卡尺，以倾向性评分标准差的倍数表示
            ratio: 匹配比例（对照组:处理组）
            replace: 是否允许放回匹配
        """
        self.caliper = caliper
        self.ratio = ratio
        self.replace = replace
        self.ps_model = None
        self.scaler = StandardScaler()
        self.matched_pairs = None
        self.balance_check = None
        
    def estimate_propensity_scores(self, 
                                   df: pd.DataFrame,
                                   treatment_col: str,
                                   feature_cols: List[str]) -> np.ndarray:
        """
        估计倾向性评分
        
        Args:
            df: 包含处理组标识和特征的DataFrame
            treatment_col: 处理组标识列名（1=处理组，0=对照组）
            feature_cols: 用于匹配的特征列名列表
            
        Returns:
            倾向性评分数组
        """
        logger.info("Estimating propensity scores...")
        
        X = df[feature_cols].copy()
        y = df[treatment_col].copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练逻辑回归模型
        self.ps_model = LogisticRegression(random_state=42, max_iter=1000)
        self.ps_model.fit(X_scaled, y)
        
        # 预测倾向性评分
        propensity_scores = self.ps_model.predict_proba(X_scaled)[:, 1]
        
        logger.info(f"Propensity scores estimated, range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")
        
        return propensity_scores
    
    def match(self, 
              df: pd.DataFrame,
              treatment_col: str,
              feature_cols: List[str],
              propensity_scores: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        执行倾向性评分匹配
        
        Args:
            df: 原始数据
            treatment_col: 处理组标识列
            feature_cols: 特征列
            propensity_scores: 预计算的倾向性评分（可选）
            
        Returns:
            匹配后的DataFrame
        """
        logger.info("Performing propensity score matching...")
        
        # 分离处理组和对照组
        treated = df[df[treatment_col] == 1].copy()
        control = df[df[treatment_col] == 0].copy()
        
        logger.info(f"Treated group size: {len(treated)}, Control group size: {len(control)}")
        
        # 计算或使用倾向性评分
        if propensity_scores is None:
            ps = self.estimate_propensity_scores(df, treatment_col, feature_cols)
            treated['ps'] = ps[df[treatment_col] == 1]
            control['ps'] = ps[df[treatment_col] == 0]
        else:
            treated['ps'] = propensity_scores[df[treatment_col] == 1]
            control['ps'] = propensity_scores[df[treatment_col] == 0]
        
        # 标准化倾向性评分
        ps_scaler = StandardScaler()
        treated['ps_std'] = ps_scaler.fit_transform(treated[['ps']])
        control['ps_std'] = ps_scaler.transform(control[['ps']])
        
        # 计算卡尺阈值
        ps_std = np.concatenate([treated['ps_std'].values, control['ps_std'].values])
        caliper_value = self.caliper * ps_std.std()
        
        # 使用KNN进行匹配
        knn = NearestNeighbors(n_neighbors=self.ratio, metric='euclidean')
        knn.fit(control[['ps_std']].values)
        
        distances, indices = knn.kneighbors(treated[['ps_std']].values)
        
        # 应用卡尺限制
        valid_matches = distances <= caliper_value
        
        # 构建匹配结果
        matched_pairs = []
        used_controls = set()
        
        for i, (treated_idx, control_indices) in enumerate(zip(treated.index, indices)):
            for j, control_idx in enumerate(control_indices):
                if valid_matches[i, j]:
                    if not self.replace and control_idx in used_controls:
                        continue
                    
                    matched_pairs.append({
                        'treated_id': treated_idx,
                        'control_id': control_idx,
                        'distance': distances[i, j],
                        'treated_ps': treated.loc[treated_idx, 'ps'],
                        'control_ps': control.loc[control_idx, 'ps']
                    })
                    
                    if not self.replace:
                        used_controls.add(control_idx)
                    
                    break  # 只取最近的一个匹配
        
        self.matched_pairs = pd.DataFrame(matched_pairs)
        
        logger.info(f"Matched {len(self.matched_pairs)} pairs")
        
        # 构建匹配后的数据集
        matched_treated = df.loc[self.matched_pairs['treated_id']].copy()
        matched_control = df.loc[self.matched_pairs['control_id']].copy()
        
        matched_treated['matched_id'] = range(len(matched_treated))
        matched_control['matched_id'] = range(len(matched_control))
        
        matched_df = pd.concat([matched_treated, matched_control], axis=0)
        
        return matched_df
    
    def check_balance(self, 
                      df: pd.DataFrame,
                      treatment_col: str,
                      feature_cols: List[str],
                      matched_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        检查匹配后的平衡性
        
        Args:
            df: 原始数据
            treatment_col: 处理组标识列
            feature_cols: 特征列
            matched_df: 匹配后的数据（可选）
            
        Returns:
            平衡性检验结果
        """
        logger.info("Checking balance...")
        
        if matched_df is None and self.matched_pairs is not None:
            matched_treated = df.loc[self.matched_pairs['treated_id']]
            matched_control = df.loc[self.matched_pairs['control_id']]
            matched_df = pd.concat([matched_treated, matched_control], axis=0)
        
        balance_results = []
        
        for feature in feature_cols:
            # 匹配前
            before_treated = df[df[treatment_col] == 1][feature].mean()
            before_control = df[df[treatment_col] == 0][feature].mean()
            before_std = df[feature].std()
            before_smd = (before_treated - before_control) / before_std if before_std > 0 else 0
            
            # 匹配后
            after_treated = matched_df[matched_df[treatment_col] == 1][feature].mean()
            after_control = matched_df[matched_df[treatment_col] == 0][feature].mean()
            after_std = matched_df[feature].std()
            after_smd = (after_treated - after_control) / after_std if after_std > 0 else 0
            
            # 方差比
            before_var_ratio = (df[df[treatment_col] == 1][feature].var() / 
                               df[df[treatment_col] == 0][feature].var())
            after_var_ratio = (matched_df[matched_df[treatment_col] == 1][feature].var() / 
                              matched_df[matched_df[treatment_col] == 0][feature].var())
            
            balance_results.append({
                'feature': feature,
                'before_treated_mean': before_treated,
                'before_control_mean': before_control,
                'before_smd': abs(before_smd),
                'after_treated_mean': after_treated,
                'after_control_mean': after_control,
                'after_smd': abs(after_smd),
                'smd_reduction': (abs(before_smd) - abs(after_smd)) / abs(before_smd) if before_smd != 0 else 0,
                'before_var_ratio': before_var_ratio,
                'after_var_ratio': after_var_ratio,
                'is_balanced': abs(after_smd) < 0.1
            })
        
        self.balance_check = pd.DataFrame(balance_results)
        
        # 统计平衡性
        n_balanced = self.balance_check['is_balanced'].sum()
        logger.info(f"Balance check: {n_balanced}/{len(feature_cols)} features balanced (SMD < 0.1)")
        
        return self.balance_check
    
    def estimate_ate(self, 
                    df: pd.DataFrame,
                    outcome_col: str,
                    treatment_col: str,
                    matched_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        估计平均处理效应(ATE)
        
        Args:
            df: 原始数据
            outcome_col: 结果变量列名
            treatment_col: 处理组标识列
            matched_df: 匹配后的数据（可选）
            
        Returns:
            ATE估计结果
        """
        logger.info("Estimating Average Treatment Effect (ATE)...")
        
        if matched_df is None and self.matched_pairs is not None:
            matched_treated = df.loc[self.matched_pairs['treated_id']]
            matched_control = df.loc[self.matched_pairs['control_id']]
            matched_df = pd.concat([matched_treated, matched_control], axis=0)
        
        # 计算ATE
        treated_outcome = matched_df[matched_df[treatment_col] == 1][outcome_col].mean()
        control_outcome = matched_df[matched_df[treatment_col] == 0][outcome_col].mean()
        ate = treated_outcome - control_outcome
        
        # 计算标准误（使用配对t检验）
        if 'matched_id' in matched_df.columns:
            # 如果有匹配ID，使用配对t检验
            paired_data = matched_df.pivot(index='matched_id', columns=treatment_col, values=outcome_col)
            paired_data.columns = ['control', 'treated']
            paired_data = paired_data.dropna()
            
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(paired_data['treated'], paired_data['control'])
            se = paired_data['treated'].std() / np.sqrt(len(paired_data))
        else:
            # 否则使用独立样本t检验
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(
                matched_df[matched_df[treatment_col] == 1][outcome_col],
                matched_df[matched_df[treatment_col] == 0][outcome_col]
            )
            se = matched_df[outcome_col].std() / np.sqrt(len(matched_df))
        
        # 计算置信区间
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        result = {
            'ate': ate,
            'treated_mean': treated_outcome,
            'control_mean': control_outcome,
            'se': se,
            't_stat': t_stat,
            'p_value': p_value,
            'ci_95': (ci_lower, ci_upper),
            'is_significant': p_value < 0.05,
            'sample_size': len(matched_df) // 2
        }
        
        logger.info(f"ATE = {ate:.4f}, p-value = {p_value:.4f}, significant: {result['is_significant']}")
        
        return result
    
    def estimate_att(self,
                    df: pd.DataFrame,
                    outcome_col: str,
                    treatment_col: str,
                    matched_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        估计处理组平均处理效应(ATT)
        
        Args:
            df: 原始数据
            outcome_col: 结果变量列名
            treatment_col: 处理组标识列
            matched_df: 匹配后的数据（可选）
            
        Returns:
            ATT估计结果
        """
        logger.info("Estimating Average Treatment Effect on Treated (ATT)...")
        
        if matched_df is None and self.matched_pairs is not None:
            matched_treated = df.loc[self.matched_pairs['treated_id']]
            matched_control = df.loc[self.matched_pairs['control_id']]
            matched_df = pd.concat([matched_treated, matched_control], axis=0)
        
        # ATT就是匹配后处理组和对照组的均值差
        treated_outcome = matched_df[matched_df[treatment_col] == 1][outcome_col].mean()
        control_outcome = matched_df[matched_df[treatment_col] == 0][outcome_col].mean()
        att = treated_outcome - control_outcome
        
        # 计算标准误
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            matched_df[matched_df[treatment_col] == 1][outcome_col],
            matched_df[matched_df[treatment_col] == 0][outcome_col]
        )
        se = matched_df[outcome_col].std() / np.sqrt(len(matched_df))
        
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se
        
        result = {
            'att': att,
            'treated_mean': treated_outcome,
            'control_mean': control_outcome,
            'se': se,
            't_stat': t_stat,
            'p_value': p_value,
            'ci_95': (ci_lower, ci_upper),
            'is_significant': p_value < 0.05,
            'sample_size': len(matched_df) // 2
        }
        
        logger.info(f"ATT = {att:.4f}, p-value = {p_value:.4f}, significant: {result['is_significant']}")
        
        return result


class DifferenceInDifferences:
    """
    双重差分(DID)模型实现
    """
    
    def __init__(self):
        self.model = None
        self.results = None
        
    def fit(self,
            df: pd.DataFrame,
            outcome_col: str,
            treatment_col: str,
            time_col: str,
            covariates: Optional[List[str]] = None) -> Dict:
        """
        拟合DID模型
        
        Args:
            df: 面板数据
            outcome_col: 结果变量
            treatment_col: 处理组标识（1=处理组，0=对照组）
            time_col: 时间标识（1=处理后，0=处理前）
            covariates: 协变量列表（可选）
            
        Returns:
            DID估计结果
        """
        logger.info("Fitting Difference-in-Differences model...")
        
        # 构建交互项
        df = df.copy()
        df['treatment_time'] = df[treatment_col] * df[time_col]
        
        # 构建公式
        formula = f"{outcome_col} ~ {treatment_col} + {time_col} + treatment_time"
        
        if covariates:
            formula += " + " + " + ".join(covariates)
        
        # 拟合OLS模型
        self.model = ols(formula, data=df).fit()
        
        # 提取结果
        self.results = {
            'did_estimator': self.model.params['treatment_time'],
            'p_value': self.model.pvalues['treatment_time'],
            'conf_int': self.model.conf_int().loc['treatment_time'].tolist(),
            'r_squared': self.model.rsquared,
            'adj_r_squared': self.model.rsquared_adj,
            'f_statistic': self.model.fvalue,
            'f_pvalue': self.model.f_pvalue,
            'sample_size': len(df)
        }
        
        # 计算各组的均值
        means = df.groupby([treatment_col, time_col])[outcome_col].mean().unstack()
        
        self.results.update({
            'control_before': means.loc[0, 0] if 0 in means.index and 0 in means.columns else None,
            'control_after': means.loc[0, 1] if 0 in means.index and 1 in means.columns else None,
            'treatment_before': means.loc[1, 0] if 1 in means.index and 0 in means.columns else None,
            'treatment_after': means.loc[1, 1] if 1 in means.index and 1 in means.columns else None
        })
        
        # 计算平行趋势检验（如果有多个时间点）
        self.results['is_significant'] = self.results['p_value'] < 0.05
        
        logger.info(f"DID estimator = {self.results['did_estimator']:.4f}, p-value = {self.results['p_value']:.4f}")
        
        return self.results
    
    def parallel_trends_test(self,
                            df: pd.DataFrame,
                            outcome_col: str,
                            treatment_col: str,
                            time_col: str,
                            pre_periods: List) -> Dict:
        """
        平行趋势假设检验
        
        Args:
            df: 面板数据
            outcome_col: 结果变量
            treatment_col: 处理组标识
            time_col: 时间列（多个时间点）
            pre_periods: 处理前的时间点列表
            
        Returns:
            平行趋势检验结果
        """
        logger.info("Testing parallel trends assumption...")
        
        # 只使用处理前数据
        pre_data = df[df[time_col].isin(pre_periods)].copy()
        
        # 构建时间趋势与处理组的交互项
        pre_data['time_trend'] = pre_data[time_col].astype(float)
        pre_data['treatment_trend'] = pre_data[treatment_col] * pre_data['time_trend']
        
        # 拟合模型
        formula = f"{outcome_col} ~ {treatment_col} + time_trend + treatment_trend"
        model = ols(formula, data=pre_data).fit()
        
        # 检验交互项是否显著（如果显著，则平行趋势假设可能不成立）
        p_value = model.pvalues['treatment_trend']
        coef = model.params['treatment_trend']
        
        result = {
            'trend_difference': coef,
            'p_value': p_value,
            'parallel_trends_assumption_holds': p_value > 0.05,
            'model_summary': model.summary().as_text()
        }
        
        logger.info(f"Parallel trends test p-value: {p_value:.4f}, holds: {result['parallel_trends_assumption_holds']}")
        
        return result
    
    def placebo_test(self,
                    df: pd.DataFrame,
                    outcome_col: str,
                    treatment_col: str,
                    time_col: str,
                    placebo_time: str) -> Dict:
        """
        安慰剂检验（假设处理发生在更早的时间）
        
        Args:
            df: 面板数据
            outcome_col: 结果变量
            treatment_col: 处理组标识
            time_col: 时间列
            placebo_time: 安慰剂处理时间点
            
        Returns:
            安慰剂检验结果
        """
        logger.info(f"Running placebo test with treatment at {placebo_time}...")
        
        # 创建安慰剂时间标识
        df = df.copy()
        df['placebo_time'] = (df[time_col] >= placebo_time).astype(int)
        df['placebo_interaction'] = df[treatment_col] * df['placebo_time']
        
        # 只使用实际处理前的数据
        actual_treatment_time = df[time_col].max()  # 假设最后一个是实际处理时间
        pre_data = df[df[time_col] < actual_treatment_time].copy()
        
        # 拟合模型
        formula = f"{outcome_col} ~ {treatment_col} + placebo_time + placebo_interaction"
        model = ols(formula, data=pre_data).fit()
        
        # 如果安慰剂效应显著，说明存在其他混杂因素
        placebo_effect = model.params['placebo_interaction']
        p_value = model.pvalues['placebo_interaction']
        
        result = {
            'placebo_effect': placebo_effect,
            'p_value': p_value,
            'is_robust': p_value > 0.05,  # 安慰剂效应不显著，说明结果稳健
            'model_summary': model.summary().as_text()
        }
        
        logger.info(f"Placebo effect p-value: {p_value:.4f}, robust: {result['is_robust']}")
        
        return result


class CausalInferencePipeline:
    """
    因果推断完整管道
    """
    
    def __init__(self):
        self.psm = PropensityScoreMatcher()
        self.did = DifferenceInDifferences()
        self.results = {}
        
    def run_analysis(self,
                    df: pd.DataFrame,
                    treatment_col: str,
                    outcome_col: str,
                    feature_cols: List[str],
                    time_col: Optional[str] = None,
                    panel_data: bool = False) -> Dict:
        """
        运行完整的因果推断分析
        
        Args:
            df: 数据
            treatment_col: 处理组标识
            outcome_col: 结果变量
            feature_cols: 特征列（用于PSM）
            time_col: 时间列（用于DID）
            panel_data: 是否为面板数据
            
        Returns:
            完整的分析结果
        """
        logger.info("Running complete causal inference pipeline...")
        
        results = {}
        
        # 1. PSM分析
        logger.info("\n=== Step 1: Propensity Score Matching ===")
        
        # 估计倾向性评分
        ps = self.psm.estimate_propensity_scores(df, treatment_col, feature_cols)
        
        # 执行匹配
        matched_df = self.psm.match(df, treatment_col, feature_cols, ps)
        
        # 平衡性检验
        balance = self.psm.check_balance(df, treatment_col, feature_cols, matched_df)
        results['balance_check'] = balance.to_dict('records')
        
        # 估计ATE
        ate = self.psm.estimate_ate(df, outcome_col, treatment_col, matched_df)
        results['psm_ate'] = ate
        
        # 2. DID分析（如果是面板数据）
        if panel_data and time_col is not None:
            logger.info("\n=== Step 2: Difference-in-Differences ===")
            
            did_results = self.did.fit(df, outcome_col, treatment_col, time_col)
            results['did'] = did_results
            
            # 平行趋势检验
            pre_periods = sorted(df[time_col].unique())[:-1]  # 假设最后一个时间是处理后
            parallel_test = self.did.parallel_trends_test(
                df, outcome_col, treatment_col, time_col, pre_periods
            )
            results['parallel_trends_test'] = parallel_test
            
            # 安慰剂检验
            if len(pre_periods) > 1:
                placebo_time = pre_periods[-1]  # 用处理前最后一个时间点做安慰剂
                placebo_test = self.did.placebo_test(
                    df, outcome_col, treatment_col, time_col, placebo_time
                )
                results['placebo_test'] = placebo_test
        
        # 3. 总结
        results['summary'] = self._generate_summary(results)
        
        self.results = results
        return results
    
    def _generate_summary(self, results: Dict) -> str:
        """生成分析总结"""
        summary = []
        summary.append("=" * 50)
        summary.append("CAUSAL INFERENCE ANALYSIS SUMMARY")
        summary.append("=" * 50)
        
        if 'psm_ate' in results:
            ate = results['psm_ate']
            summary.append("\n📊 Propensity Score Matching Results:")
            summary.append(f"  - ATE: {ate['ate']:.4f}")
            summary.append(f"  - 95% CI: [{ate['ci_95'][0]:.4f}, {ate['ci_95'][1]:.4f}]")
            summary.append(f"  - p-value: {ate['p_value']:.4f}")
            summary.append(f"  - Significant: {ate['is_significant']}")
        
        if 'did' in results:
            did = results['did']
            summary.append("\n📈 Difference-in-Differences Results:")
            summary.append(f"  - DID Estimator: {did['did_estimator']:.4f}")
            summary.append(f"  - p-value: {did['p_value']:.4f}")
            summary.append(f"  - R-squared: {did['r_squared']:.4f}")
            summary.append(f"  - Significant: {did['is_significant']}")
        
        if 'parallel_trends_test' in results:
            pt = results['parallel_trends_test']
            summary.append(f"\n🔄 Parallel Trends Test:")
            summary.append(f"  - Holds: {pt['parallel_trends_assumption_holds']}")
        
        summary.append("\n" + "=" * 50)
        
        return "\n".join(summary)


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    
    # 特征
    df = pd.DataFrame({
        'user_id': range(n),
        'age': np.random.normal(35, 10, n),
        'investment_exp': np.random.exponential(5, n),
        'risk_score': np.random.uniform(0, 10, n),
        'query_frequency': np.random.poisson(10, n),
        'treatment': np.random.binomial(1, 0.3, n),  # 处理组
        'satisfaction': np.random.normal(3.5, 1, n),  # 结果变量
        'time': np.random.choice([0, 1], n)  # 时间（0=前，1=后）
    })
    
    # 添加处理效应（假设处理组满意度+0.5）
    df.loc[df['treatment'] == 1, 'satisfaction'] += 0.5
    
    # 特征列
    feature_cols = ['age', 'investment_exp', 'risk_score', 'query_frequency']
    
    # 运行因果推断
    pipeline = CausalInferencePipeline()
    results = pipeline.run_analysis(
        df=df,
        treatment_col='treatment',
        outcome_col='satisfaction',
        feature_cols=feature_cols,
        time_col='time',
        panel_data=True
    )
    
    # 打印结果
    print(results['summary'])
    
    # 查看平衡性检验
    if 'balance_check' in results:
        balance_df = pd.DataFrame(results['balance_check'])
        print("\n📊 Balance Check (first 5 features):")
        print(balance_df[['feature', 'before_smd', 'after_smd', 'is_balanced']].head())
        