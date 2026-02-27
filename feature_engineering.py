"""
特征工程模块：构建用户、查询、回答等特征
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def build_user_features(self, users_df: pd.DataFrame, 
                           queries_df: pd.DataFrame,
                           feedback_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建用户特征
        
        Args:
            users_df: 用户基础信息表
            queries_df: 用户查询记录表
            feedback_df: 用户反馈表
            
        Returns:
            用户特征DataFrame
        """
        logger.info("Building user features...")
        
        # 基础特征
        user_features = users_df.copy()
        
        # 处理分类变量
        categorical_cols = ['user_type', 'risk_profile']
        for col in categorical_cols:
            if col in user_features.columns:
                self.label_encoders[col] = LabelEncoder()
                user_features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    user_features[col].fillna('unknown')
                )
        
        # 时间特征
        if 'registration_date' in user_features.columns:
            user_features['registration_date'] = pd.to_datetime(user_features['registration_date'])
            user_features['reg_days'] = (datetime.now() - user_features['registration_date']).dt.days
        
        # 聚合查询特征
        if not queries_df.empty:
            # 查询频率
            query_agg = queries_df.groupby('user_id').agg({
                'query_id': 'count',
                'query_time': lambda x: (x.max() - x.min()).days if len(x) > 1 else 0
            }).rename(columns={
                'query_id': 'total_queries',
                'query_time': 'query_span_days'
            })
            user_features = user_features.merge(query_agg, on='user_id', how='left')
            
            # 查询类型分布
            query_type_pivot = pd.crosstab(
                queries_df['user_id'], 
                queries_df['query_type'],
                normalize='index'
            ).add_prefix('query_type_')
            user_features = user_features.merge(
                query_type_pivot, 
                on='user_id', 
                how='left'
            )
        
        # 聚合反馈特征
        if not feedback_df.empty and not queries_df.empty:
            # 关联查询和反馈
            q_with_feedback = queries_df.merge(
                feedback_df[['query_id', 'rating', 'nps_score']],
                on='query_id',
                how='inner'
            )
            
            feedback_agg = q_with_feedback.groupby('user_id').agg({
                'rating': ['mean', 'std', 'count'],
                'nps_score': 'mean'
            })
            feedback_agg.columns = ['avg_rating', 'std_rating', 'feedback_count', 'avg_nps']
            user_features = user_features.merge(feedback_agg, on='user_id', how='left')
        
        # 填充缺失值
        user_features = user_features.fillna(0)
        
        logger.info(f"User features shape: {user_features.shape}")
        return user_features
    
    def build_query_features(self, queries_df: pd.DataFrame,
                            stocks_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建查询特征
        
        Args:
            queries_df: 查询记录表
            stocks_df: 股票信息表
            
        Returns:
            查询特征DataFrame
        """
        logger.info("Building query features...")
        
        query_features = queries_df.copy()
        
        # 时间特征
        query_features['query_time'] = pd.to_datetime(query_features['query_time'])
        query_features['hour'] = query_features['query_time'].dt.hour
        query_features['day_of_week'] = query_features['query_time'].dt.dayofweek
        query_features['is_weekend'] = query_features['day_of_week'].isin([5, 6]).astype(int)
        query_features['is_trading_hour'] = (
            (query_features['hour'] >= 9) & 
            (query_features['hour'] <= 15)
        ).astype(int)
        
        # 查询文本特征
        if 'query_text' in query_features.columns:
            query_features['query_length'] = query_features['query_text'].str.len()
            query_features['word_count'] = query_features['query_text'].str.split().str.len()
            
            # 关键词匹配
            finance_keywords = ['财报', 'PE', '估值', '增长率', '毛利率', 'ROE', '现金流']
            for kw in finance_keywords:
                query_features[f'has_{kw}'] = query_features['query_text'].str.contains(
                    kw, na=False
                ).astype(int)
        
        # 股票相关特征
        if stocks_df is not None and not stocks_df.empty:
            # 解析查询中的股票代码
            stock_features = []
            for idx, row in query_features.iterrows():
                if row.get('stock_codes'):
                    try:
                        codes = json.loads(row['stock_codes']) if isinstance(row['stock_codes'], str) else row['stock_codes']
                        if codes:
                            stock_info = stocks_df[stocks_df['stock_code'].isin(codes)]
                            stock_features.append({
                                'query_id': row['query_id'],
                                'n_stocks': len(codes),
                                'avg_market_cap': stock_info['market_cap'].mean() if not stock_info.empty else 0,
                                'industries': stock_info['industry'].nunique() if not stock_info.empty else 0
                            })
                    except:
                        pass
            
            if stock_features:
                stock_features_df = pd.DataFrame(stock_features)
                query_features = query_features.merge(
                    stock_features_df, 
                    on='query_id', 
                    how='left'
                )
        
        # 填充缺失值
        fill_cols = ['n_stocks', 'avg_market_cap', 'industries']
        for col in fill_cols:
            if col in query_features.columns:
                query_features[col] = query_features[col].fillna(0)
        
        logger.info(f"Query features shape: {query_features.shape}")
        return query_features
    
    def build_response_features(self, responses_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建模型回答特征
        
        Args:
            responses_df: 模型回答表
            
        Returns:
            回答特征DataFrame
        """
        logger.info("Building response features...")
        
        response_features = responses_df.copy()
        
        # 响应时间特征
        if 'response_time_ms' in response_features.columns:
            response_features['response_time_sec'] = response_features['response_time_ms'] / 1000
            response_features['response_time_log'] = np.log1p(response_features['response_time_ms'])
        
        # 文本特征
        if 'response_text' in response_features.columns:
            response_features['response_length'] = response_features['response_text'].str.len()
            response_features['response_paragraphs'] = response_features['response_text'].str.count('\n\n') + 1
            
            # 财务数据出现次数
            financial_patterns = ['亿元', '万元', '增长率', '利润率', 'PE', 'PB', 'ROE']
            for pattern in financial_patterns:
                response_features[f'count_{pattern}'] = response_features['response_text'].str.count(
                    pattern, na=False
                )
            
            response_features['financial_mentions'] = response_features[[
                f'count_{p}' for p in financial_patterns
            ]].sum(axis=1)
        
        # 模型元特征
        if 'model_version' in response_features.columns:
            self.label_encoders['model_version'] = LabelEncoder()
            response_features['model_version_encoded'] = self.label_encoders['model_version'].fit_transform(
                response_features['model_version'].fillna('unknown')
            )
        
        if 'retrieval_method' in response_features.columns:
            self.label_encoders['retrieval_method'] = LabelEncoder()
            response_features['retrieval_method_encoded'] = self.label_encoders['retrieval_method'].fit_transform(
                response_features['retrieval_method'].fillna('none')
            )
        
        # 填充缺失值
        response_features = response_features.fillna(0)
        
        logger.info(f"Response features shape: {response_features.shape}")
        return response_features
    
    def build_complete_feature_set(self, 
                                   users_df: pd.DataFrame,
                                   queries_df: pd.DataFrame,
                                   responses_df: pd.DataFrame,
                                   feedback_df: pd.DataFrame,
                                   stocks_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建完整特征集
        
        Returns:
            所有特征合并的DataFrame
        """
        logger.info("Building complete feature set...")
        
        # 构建各维度特征
        user_features = self.build_user_features(users_df, queries_df, feedback_df)
        query_features = self.build_query_features(queries_df, stocks_df)
        response_features = self.build_response_features(responses_df)
        
        # 合并特征
        # 先关联查询和回答
        query_response = query_features.merge(
            response_features,
            on='query_id',
            how='inner'
        )
        
        # 关联用户特征
        complete_features = query_response.merge(
            user_features,
            on='user_id',
            how='left'
        )
        
        # 关联反馈作为标签
        if not feedback_df.empty:
            complete_features = complete_features.merge(
                feedback_df[['query_id', 'rating', 'nps_score']],
                on='query_id',
                how='left'
            )
        
        # 记录特征列名
        self.feature_columns = [col for col in complete_features.columns 
                               if col not in ['user_id', 'query_id', 'response_id',
                                            'query_text', 'response_text', 
                                            'rating', 'nps_score']]
        
        logger.info(f"Complete features shape: {complete_features.shape}")
        logger.info(f"Number of features: {len(self.feature_columns)}")
        
        return complete_features
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        特征标准化
        
        Args:
            df: 特征DataFrame
            fit: 是否拟合标准化器
            
        Returns:
            标准化后的特征
        """
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols].fillna(0))
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols].fillna(0))
        
        return df
    
    def get_feature_importance(self, model, feature_names: List[str] = None) -> pd.DataFrame:
        """
        获取特征重要性（适用于树模型）
        
        Args:
            model: 训练好的模型
            feature_names: 特征名列表
            
        Returns:
            特征重要性DataFrame
        """
        if feature_names is None:
            feature_names = self.feature_columns
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()