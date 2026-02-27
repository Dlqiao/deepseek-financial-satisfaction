#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek金融垂域股票分析满意度项目 - 探索性数据分析(EDA)
完全使用Python脚本，不需要Jupyter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')


class FinancialEDA:
    """
    金融满意度数据分析类
    包含数据生成、可视化、分析报告等功能
    """
    
    def __init__(self, n_samples=10000, random_seed=42):
        """
        初始化EDA分析器
        
        Args:
            n_samples: 样本数量
            random_seed: 随机种子
        """
        self.n_samples = n_samples
        np.random.seed(random_seed)
        self.df = None
        self.stocks_df = None
        self.users_df = None
        
    def generate_data(self):
        """生成模拟数据（实际项目中应从数据库读取）"""
        print("📊 正在生成模拟数据...")
        
        # 1. 股票数据
        self.stocks_df = pd.DataFrame({
            'stock_code': ['600519.SH', '000858.SZ', '601318.SH', '600036.SH', '000002.SZ',
                          '002415.SZ', '300750.SZ', '000333.SZ', '002594.SZ', '688981.SH',
                          '300059.SZ', '600030.SH', '000001.SZ', '002714.SZ', '300760.SZ'],
            'stock_name': ['贵州茅台', '五粮液', '中国平安', '招商银行', '万科A',
                          '海康威视', '宁德时代', '美的集团', '比亚迪', '中芯国际',
                          '东方财富', '中信证券', '平安银行', '牧原股份', '迈瑞医疗'],
            'industry': ['食品饮料', '食品饮料', '保险', '银行', '房地产',
                        '计算机', '电力设备', '家电', '汽车', '电子',
                        '证券', '证券', '银行', '农林牧渔', '医药生物'],
            'market_cap': [20000, 6000, 8000, 9000, 1500,
                          3500, 12000, 4500, 7000, 4000,
                          2500, 3000, 2000, 2500, 3800],
            'pe_ttm': [30, 25, 8, 6, 10, 25, 40, 15, 60, 50, 30, 15, 5, 20, 35]
        })
        
        # 2. 用户数据
        self.users_df = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(1, 1001)],
            'user_type': np.random.choice(['免费', '付费', '企业'], size=1000, p=[0.6, 0.3, 0.1]),
            'risk_profile': np.random.choice(['保守', '稳健', '进取'], size=1000, p=[0.3, 0.5, 0.2]),
            'investment_exp': np.random.randint(0, 20, 1000),
            'registration_days': np.random.randint(1, 500, 1000)
        })
        
        # 3. 查询数据
        queries = []
        query_types = ['基本面分析', '技术分析', '财报解读', '行业对比', '投资建议', '股价查询']
        experiment_groups = ['control', 'treatment_rag', 'treatment_prompt', 'treatment_combined']
        group_weights = [0.4, 0.2, 0.2, 0.2]
        
        for i in range(self.n_samples):
            user_id = f'user_{np.random.randint(1, 1001)}'
            n_stocks = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            selected_stocks = self.stocks_df.sample(n_stocks)
            
            hour = np.random.randint(0, 24)
            is_trading_hour = (9 <= hour <= 11) or (13 <= hour <= 15)
            
            queries.append({
                'query_id': i + 10000,
                'user_id': user_id,
                'query_time': pd.Timestamp('2025-02-01') + pd.Timedelta(hours=np.random.randint(0, 720)),
                'query_type': np.random.choice(query_types),
                'stock_codes': ','.join(selected_stocks['stock_code'].tolist()),
                'stock_names': ','.join(selected_stocks['stock_name'].tolist()),
                'industries': ','.join(selected_stocks['industry'].unique()),
                'avg_market_cap': selected_stocks['market_cap'].mean(),
                'min_market_cap': selected_stocks['market_cap'].min(),
                'max_market_cap': selected_stocks['market_cap'].max(),
                'n_stocks': n_stocks,
                'hour': hour,
                'is_trading_hour': is_trading_hour,
                'day_of_week': np.random.randint(0, 7),
                'experiment_group': np.random.choice(experiment_groups, p=group_weights),
                'query_length': np.random.randint(20, 200)
            })
        
        queries_df = pd.DataFrame(queries)
        
        # 4. 反馈数据
        feedback = []
        for _, query in queries_df.iterrows():
            # 基础满意度（大盘股满意度高，小盘股满意度低）
            base_satisfaction = 0.5
            
            if query['avg_market_cap'] > 1000:
                base_satisfaction += 0.3
            elif query['avg_market_cap'] > 100:
                base_satisfaction += 0.1
            elif query['avg_market_cap'] > 10:
                base_satisfaction -= 0.1
            else:
                base_satisfaction -= 0.3
            
            # 实验组影响
            if query['experiment_group'] == 'treatment_rag':
                base_satisfaction += 0.15
            elif query['experiment_group'] == 'treatment_prompt':
                base_satisfaction += 0.1
            elif query['experiment_group'] == 'treatment_combined':
                base_satisfaction += 0.25
            
            # 查询类型影响
            if query['query_type'] in ['财报解读', '基本面分析']:
                base_satisfaction += 0.05
            
            prob = 1 / (1 + np.exp(-base_satisfaction))
            is_satisfied = np.random.random() < prob
            
            if is_satisfied:
                rating = np.random.randint(4, 6)
                nps = np.random.randint(7, 11)
            else:
                rating = np.random.randint(1, 4)
                nps = np.random.randint(0, 7)
            
            feedback.append({
                'query_id': query['query_id'],
                'rating': rating,
                'nps_score': nps,
                'is_satisfied': 1 if rating >= 4 else 0
            })
        
        feedback_df = pd.DataFrame(feedback)
        
        # 5. 合并数据
        self.df = queries_df.merge(feedback_df, on='query_id')
        self.df = self.df.merge(self.users_df, on='user_id')
        
        # 添加市值分类
        def categorize_market_cap(cap):
            if cap > 1000:
                return '大盘股 (>1000亿)'
            elif cap > 100:
                return '中盘股 (100-1000亿)'
            elif cap > 10:
                return '小盘股 (10-100亿)'
            else:
                return '微盘股 (<10亿)'
        
        self.df['market_cap_category'] = self.df['avg_market_cap'].apply(categorize_market_cap)
        self.df['hour'] = self.df['query_time'].dt.hour
        
        print(f"✅ 数据生成完成：{len(self.df)} 条记录")
        print(f"   - 用户数: {self.df['user_id'].nunique()}")
        print(f"   - 时间范围: {self.df['query_time'].min()} 到 {self.df['query_time'].max()}")
        
        return self.df
    
    def data_overview(self):
        """数据概览"""
        print("\n" + "="*60)
        print("📋 数据概览")
        print("="*60)
        
        print(f"总记录数: {len(self.df):,}")
        print(f"唯一用户数: {self.df['user_id'].nunique():,}")
        print(f"时间范围: {self.df['query_time'].min()} 到 {self.df['query_time'].max()}")
        
        print("\n核心指标统计:")
        print(self.df[['rating', 'nps_score', 'is_satisfied']].describe())
        
        print("\n缺失值检查:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if any(missing > 0) else "无缺失值")
    
    def plot_satisfaction_distribution(self, save_path=None):
        """满意度分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 评分分布
        axes[0, 0].hist(self.df['rating'], bins=5, edgecolor='black', color='skyblue')
        axes[0, 0].set_xlabel('评分')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('用户评分分布')
        
        # 2. NPS分布
        axes[0, 1].hist(self.df['nps_score'], bins=11, edgecolor='black', color='lightgreen')
        axes[0, 1].set_xlabel('NPS评分')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('NPS评分分布')
        
        # 3. 用户类型分布
        user_type_counts = self.df['user_type'].value_counts()
        axes[1, 0].pie(user_type_counts.values, labels=user_type_counts.index, 
                       autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1, 0].set_title('用户类型分布')
        
        # 4. 查询类型分布
        query_type_counts = self.df['query_type'].value_counts()
        axes[1, 1].barh(query_type_counts.index, query_type_counts.values, color='orange')
        axes[1, 1].set_xlabel('频次')
        axes[1, 1].set_title('查询类型分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 图表已保存: {save_path}")
        
        plt.show()
    
    def analyze_hunger_phenomenon(self, save_path=None):
        """
        分析"吃不饱"现象：市值与满意度的关系
        """
        print("\n" + "="*60)
        print("🍽️ 验证'吃不饱'现象：市值与满意度的关系")
        print("="*60)
        
        # 按市值分类统计
        market_stats = self.df.groupby('market_cap_category').agg({
            'is_satisfied': ['mean', 'count'],
            'rating': 'mean',
            'nps_score': 'mean'
        }).round(3)
        market_stats.columns = ['满意度', '样本量', '平均评分', '平均NPS']
        market_stats = market_stats.reset_index()
        
        print("\n不同市值股票的满意度对比:")
        print(market_stats.to_string(index=False))
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 满意度柱状图
        ax1 = axes[0]
        colors = ['green' if i == 0 else 'orange' if i == 1 else 'red' if i == 2 else 'darkred' 
                  for i in range(len(market_stats))]
        bars = ax1.bar(market_stats['market_cap_category'], market_stats['满意度'], color=colors)
        ax1.set_xlabel('股票市值分类')
        ax1.set_ylabel('满意度')
        ax1.set_title('不同市值股票的满意度对比')
        ax1.set_ylim(0, 1)
        
        for bar, val in zip(bars, market_stats['满意度']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center')
        
        # 样本量分布
        ax2 = axes[1]
        ax2.pie(market_stats['样本量'], labels=market_stats['market_cap_category'],
                autopct='%1.1f%%', colors=colors)
        ax2.set_title('各市值分类样本量分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # 行业分析
        print("\n各行业满意度排名（TOP 10）:")
        industry_stats = self.df.groupby('industries').agg({
            'is_satisfied': 'mean',
            'query_id': 'count',
            'avg_market_cap': 'mean'
        }).round(3)
        industry_stats.columns = ['满意度', '查询次数', '平均市值']
        industry_stats = industry_stats.sort_values('满意度', ascending=False).head(10)
        print(industry_stats)
        
        return market_stats
    
    def analyze_ab_test(self, save_path=None):
        """
        AB实验效果分析
        """
        print("\n" + "="*60)
        print("🧪 AB实验效果分析")
        print("="*60)
        
        # 实验组统计
        exp_stats = self.df.groupby('experiment_group').agg({
            'is_satisfied': ['mean', 'count'],
            'rating': 'mean',
            'nps_score': 'mean'
        }).round(3)
        exp_stats.columns = ['满意度', '样本量', '平均评分', '平均NPS']
        exp_stats = exp_stats.reset_index()
        
        # 分组名称映射
        group_names = {
            'control': '对照组',
            'treatment_rag': '实验组A (RAG增强)',
            'treatment_prompt': '实验组B (结构化Prompt)',
            'treatment_combined': '实验组C (RAG+Prompt组合)'
        }
        exp_stats['实验组'] = exp_stats['experiment_group'].map(group_names)
        
        print("\n各实验组成效对比:")
        print(exp_stats[['实验组', '满意度', '平均评分', '平均NPS', '样本量']].to_string(index=False))
        
        # 计算提升
        control_sat = exp_stats[exp_stats['experiment_group'] == 'control']['满意度'].values[0]
        combined_sat = exp_stats[exp_stats['experiment_group'] == 'treatment_combined']['满意度'].values[0]
        lift = (combined_sat - control_sat) / control_sat
        print(f"\n📈 实验组C相比对照组满意度提升: {lift:.1%}")
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 满意度对比
        ax1 = axes[0]
        bars1 = ax1.bar(exp_stats['实验组'], exp_stats['满意度'])
        ax1.set_xlabel('实验组')
        ax1.set_ylabel('满意度')
        ax1.set_title('各实验组满意度对比')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=15)
        
        for bar, val in zip(bars1, exp_stats['满意度']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center')
        
        # NPS对比
        ax2 = axes[1]
        bars2 = ax2.bar(exp_stats['实验组'], exp_stats['平均NPS'])
        ax2.set_xlabel('实验组')
        ax2.set_ylabel('平均NPS')
        ax2.set_title('各实验组NPS对比')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.tick_params(axis='x', rotation=15)
        
        for bar, val in zip(bars2, exp_stats['平均NPS']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # 实验组在不同市值股票上的表现
        exp_market = self.df.groupby(['experiment_group', 'market_cap_category'])['is_satisfied'].mean().unstack()
        
        plt.figure(figsize=(12, 6))
        exp_market.T.plot(kind='bar', ax=plt.gca())
        plt.xlabel('股票市值分类')
        plt.ylabel('满意度')
        plt.title('各实验组在不同市值股票上的满意度表现')
        plt.legend(title='实验组', labels=['对照组', 'RAG增强', '结构化Prompt', 'RAG+Prompt组合'])
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return exp_stats
    
    def analyze_user_behavior(self, save_path=None):
        """
        用户行为分析
        """
        print("\n" + "="*60)
        print("👥 用户行为分析")
        print("="*60)
        
        # 用户类型分析
        user_stats = self.df.groupby('user_type').agg({
            'is_satisfied': 'mean',
            'rating': 'mean',
            'nps_score': 'mean',
            'user_id': 'nunique'
        }).round(3)
        user_stats.columns = ['满意度', '平均评分', '平均NPS', '用户数']
        
        print("\n不同用户类型的满意度:")
        print(user_stats)
        
        # 查询时间分析
        hourly_stats = self.df.groupby('hour').agg({
            'is_satisfied': 'mean',
            'query_id': 'count'
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        # 柱状图：查询量
        bars = ax1.bar(hourly_stats['hour'], hourly_stats['query_id'], alpha=0.5, color='gray')
        ax1.set_xlabel('小时')
        ax1.set_ylabel('查询次数', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')
        
        # 折线图：满意度
        ax2 = ax1.twinx()
        ax2.plot(hourly_stats['hour'], hourly_stats['is_satisfied'], 'r-', linewidth=2, marker='o')
        ax2.set_ylabel('满意度', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
        
        # 标记交易时段
        ax1.axvspan(9, 11, alpha=0.2, color='green', label='交易时段 9:00-11:30')
        ax1.axvspan(13, 15, alpha=0.2, color='green')
        
        plt.title('查询时间分布与满意度变化')
        fig.legend(loc='upper right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return user_stats
    
    def correlation_analysis(self):
        """
        相关性分析
        """
        print("\n" + "="*60)
        print("🔗 相关性分析")
        print("="*60)
        
        numeric_cols = ['rating', 'nps_score', 'is_satisfied', 'avg_market_cap', 
                       'n_stocks', 'hour', 'query_length', 'investment_exp', 
                       'registration_days']
        
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                    square=True, linewidths=1, cbar_kws={'shrink': 0.8})
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.show()
        
        # 找出与满意度最相关的特征
        sat_corr = corr_matrix['is_satisfied'].drop('is_satisfied').sort_values(ascending=False)
        print("\n与满意度最相关的特征:")
        for feat, corr in sat_corr.items():
            print(f"  {feat}: {corr:.3f}")
        
        return corr_matrix
    
    def generate_full_report(self, output_dir='./reports'):
        """
        生成完整分析报告
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("📑 生成完整分析报告")
        print("="*60)
        
        # 1. 数据概览
        self.data_overview()
        
        # 2. 保存图表
        self.plot_satisfaction_distribution(save_path=f"{output_dir}/satisfaction_distribution.png")
        market_stats = self.analyze_hunger_phenomenon(save_path=f"{output_dir}/market_cap_analysis.png")
        exp_stats = self.analyze_ab_test(save_path=f"{output_dir}/ab_test_results.png")
        user_stats = self.analyze_user_behavior(save_path=f"{output_dir}/user_behavior.png")
        corr_matrix = self.correlation_analysis()
        
        # 3. 生成文本报告
        report_path = f"{output_dir}/eda_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("DEEPSEEK金融垂域满意度分析报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"样本数量: {len(self.df):,}\n")
            f.write(f"用户数量: {self.df['user_id'].nunique():,}\n\n")
            
            f.write("核心发现:\n")
            f.write("1. '吃不饱'现象验证: 大盘股满意度 {:.1%}, 小盘股满意度 {:.1%}\n".format(
                market_stats[market_stats['market_cap_category'] == '大盘股 (>1000亿)']['满意度'].values[0],
                market_stats[market_stats['market_cap_category'] == '小盘股 (10-100亿)']['满意度'].values[0]
            ))
            
            control_sat = exp_stats[exp_stats['experiment_group'] == 'control']['满意度'].values[0]
            combined_sat = exp_stats[exp_stats['experiment_group'] == 'treatment_combined']['满意度'].values[0]
            lift = (combined_sat - control_sat) / control_sat
            f.write(f"2. AB实验效果: 实验组C相比对照组提升 {lift:.1%}\n")
            
            f.write("3. 最佳用户群体: {} 用户满意度最高 ({:.1%})\n".format(
                user_stats['满意度'].idxmax(),
                user_stats['满意度'].max()
            ))
        
        print(f"✅ 报告已生成: {report_path}")
        print(f"✅ 图表已保存至: {output_dir}")
        
        return {
            'market_stats': market_stats,
            'exp_stats': exp_stats,
            'user_stats': user_stats,
            'corr_matrix': corr_matrix
        }


def main():
    """
    主函数：运行完整的EDA分析
    """
    print("="*60)
    print("🚀 开始 DeepSeek 金融垂域满意度 EDA 分析")
    print("="*60)
    
    # 初始化分析器
    eda = FinancialEDA(n_samples=10000)
    
    # 生成数据
    df = eda.generate_data()
    
    # 运行分析
    results = eda.generate_full_report(output_dir='./reports')
    
    print("\n" + "="*60)
    print("✅ EDA分析完成！")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()