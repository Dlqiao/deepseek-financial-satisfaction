"""
满意度预测模型：用户满意度预测、特征重要性分析、模型解释
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatisfactionPredictor:
    """
    用户满意度预测模型
    """
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        初始化预测器
        
        Args:
            model_type: 模型类型 ('xgboost', 'lightgbm', 'random_forest', 'logistic')
            random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.metrics = {}
        self.feature_names = None
        
    def _init_model(self, **params):
        """初始化模型"""
        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            default_params.update(params)
            self.model = xgb.XGBClassifier(**default_params)
            
        elif self.model_type == 'lightgbm':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbosity': -1
            }
            default_params.update(params)
            self.model = lgb.LGBMClassifier(**default_params)
            
        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            }
            default_params.update(params)
            self.model = RandomForestClassifier(**default_params)
            
        elif self.model_type == 'logistic':
            default_params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': self.random_state
            }
            default_params.update(params)
            self.model = LogisticRegression(**default_params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_features(self, 
                        df: pd.DataFrame,
                        feature_cols: List[str],
                        target_col: str,
                        test_size: float = 0.2,
                        validate: bool = True) -> Dict:
        """
        准备特征和标签
        
        Args:
            df: 输入数据
            feature_cols: 特征列名列表
            target_col: 目标列名
            test_size: 测试集比例
            validate: 是否进行数据验证
            
        Returns:
            包含训练集和测试集的字典
        """
        logger.info("Preparing features and labels...")
        
        # 验证数据
        if validate:
            self._validate_data(df, feature_cols, target_col)
        
        # 分离特征和标签
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 处理分类变量
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
        
        # 保存特征名
        self.feature_names = feature_cols
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 转换为DataFrame保持特征名
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
        
        logger.info(f"Training set size: {len(X_train_scaled)}, Test set size: {len(X_test_scaled)}")
        logger.info(f"Positive class ratio: {y.mean():.2%}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def _validate_data(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        """验证数据质量"""
        # 检查缺失列
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # 检查目标变量
        if df[target_col].nunique() < 2:
            raise ValueError("Target variable has only one unique value")
        
        # 检查特征方差
        for col in feature_cols:
            if df[col].nunique() == 1:
                logger.warning(f"Feature {col} has constant value, may not be useful")
    
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             **model_params) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            model_params: 模型参数
            
        Returns:
            训练历史
        """
        logger.info(f"Training {self.model_type} model...")
        
        # 初始化模型
        self._init_model(**model_params)
        
        # 训练模型
        if X_val is not None and y_val is not None:
            # 有验证集
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            # 无验证集
            self.model.fit(X_train, y_train)
        
        # 计算训练集指标
        train_pred = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        
        train_metrics = {
            'accuracy': accuracy_score(y_train, train_pred),
            'precision': precision_score(y_train, train_pred, zero_division=0),
            'recall': recall_score(y_train, train_pred, zero_division=0),
            'f1': f1_score(y_train, train_pred, zero_division=0),
            'auc': roc_auc_score(y_train, train_proba)
        }
        
        logger.info(f"Training metrics: {train_metrics}")
        
        return {'train_metrics': train_metrics}
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            评估指标
        """
        logger.info("Evaluating model...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Test metrics: {self.metrics}")
        
        return self.metrics
    
    def cross_validate(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       cv: int = 5,
                       **model_params) -> Dict:
        """
        交叉验证
        
        Args:
            X: 特征
            y: 标签
            cv: 交叉验证折数
            model_params: 模型参数
            
        Returns:
            交叉验证结果
        """
        logger.info(f"Performing {cv}-fold cross validation...")
        
        # 初始化模型
        self._init_model(**model_params)
        
        # 执行交叉验证
        cv_scores = {
            'accuracy': cross_val_score(self.model, X, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(self.model, X, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(self.model, X, y, cv=cv, scoring='recall'),
            'f1': cross_val_score(self.model, X, y, cv=cv, scoring='f1'),
            'auc': cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
        }
        
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            logger.info(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             param_grid: Dict,
                             cv: int = 5,
                             scoring: str = 'roc_auc') -> Dict:
        """
        超参数调优
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数
            scoring: 评分指标
            
        Returns:
            最佳参数和分数
        """
        logger.info(f"Performing hyperparameter tuning with {cv}-fold CV...")
        
        # 初始化基础模型
        self._init_model()
        
        # 网格搜索
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # 保存最佳模型
        self.model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best {scoring}: {grid_search.best_score_:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名列表
            
        Returns:
            特征重要性DataFrame
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        if hasattr(self.model, 'feature_importances_'):
            # 树模型
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # 线性模型
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model does not provide feature importance")
            return pd.DataFrame()
        
        # 创建DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征
            
        Returns:
            预测类别
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征
            
        Returns:
            预测概率
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data.get('metrics', {})
        logger.info(f"Model loaded from {filepath}")


class SatisfactionAnalyzer:
    """
    满意度分析器：群体分析、特征分析、阈值优化
    """
    
    def __init__(self, predictor: SatisfactionPredictor):
        """
        初始化分析器
        
        Args:
            predictor: 训练好的预测器
        """
        self.predictor = predictor
        self.analysis_results = {}
    
    def analyze_by_segments(self,
                           df: pd.DataFrame,
                           segment_cols: List[str],
                           target_col: str,
                           pred_col: str = 'predicted_satisfaction') -> Dict:
        """
        按群体分析满意度
        
        Args:
            df: 数据
            segment_cols: 分群列名
            target_col: 真实标签列
            pred_col: 预测标签列
            
        Returns:
            群体分析结果
        """
        logger.info("Analyzing satisfaction by segments...")
        
        segment_results = {}
        
        for col in segment_cols:
            if col not in df.columns:
                continue
            
            # 计算各群体的指标
            segment_analysis = df.groupby(col).agg({
                target_col: ['mean', 'count'],
                pred_col: 'mean'
            }).round(4)
            
            segment_analysis.columns = ['actual_satisfaction', 'count', 'predicted_satisfaction']
            segment_analysis['error'] = abs(segment_analysis['actual_satisfaction'] - 
                                           segment_analysis['predicted_satisfaction'])
            
            segment_results[col] = segment_analysis.to_dict('index')
        
        self.analysis_results['segment_analysis'] = segment_results
        
        return segment_results
    
    def find_optimal_threshold(self,
                              y_true: pd.Series,
                              y_proba: np.ndarray,
                              metric: str = 'f1') -> Dict:
        """
        寻找最优分类阈值
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            metric: 优化指标 ('f1', 'precision', 'recall')
            
        Returns:
            最优阈值和对应的指标
        """
        from sklearn.metrics import precision_recall_curve
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        if metric == 'f1':
            scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        elif metric == 'precision':
            scores = precisions
        elif metric == 'recall':
            scores = recalls
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # 找到最优阈值
        best_idx = np.argmax(scores[:-1])  # 排除最后一个
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        result = {
            'optimal_threshold': best_threshold,
            f'optimal_{metric}': best_score,
            'thresholds': thresholds.tolist(),
            'precisions': precisions.tolist(),
            'recalls': recalls.tolist()
        }
        
        self.analysis_results['threshold_optimization'] = result
        
        return result
    
    def analyze_errors(self,
                      df: pd.DataFrame,
                      feature_cols: List[str],
                      target_col: str,
                      pred_col: str = 'predicted') -> pd.DataFrame:
        """
        错误分析
        
        Args:
            df: 数据
            feature_cols: 特征列
            target_col: 真实标签
            pred_col: 预测标签
            
        Returns:
            错误样本分析
        """
        # 标记错误类型
        df_analysis = df.copy()
        df_analysis['error_type'] = 'correct'
        df_analysis.loc[(df_analysis[target_col] == 1) & (df_analysis[pred_col] == 0), 'error_type'] = 'false_negative'
        df_analysis.loc[(df_analysis[target_col] == 0) & (df_analysis[pred_col] == 1), 'error_type'] = 'false_positive'
        
        # 分析各特征在错误样本上的分布
        error_analysis = {}
        
        for col in feature_cols:
            if col in df_analysis.columns:
                # 计算各错误类型的特征均值
                error_stats = df_analysis.groupby('error_type')[col].agg(['mean', 'std', 'count'])
                error_analysis[col] = error_stats.to_dict()
        
        self.analysis_results['error_analysis'] = error_analysis
        
        return df_analysis
    
    def generate_insights(self) -> List[str]:
        """
        生成业务洞察
        
        Returns:
            洞察列表
        """
        insights = []
        
        # 特征重要性洞察
        if self.predictor.feature_importance is not None:
            top_features = self.predictor.feature_importance.head(3)
            insights.append(f"Top 3 important features: {', '.join(top_features['feature'].tolist())}")
        
        # 模型性能洞察
        if self.predictor.metrics:
            metrics = self.predictor.metrics
            insights.append(f"Model AUC: {metrics.get('auc', 0):.3f}")
            insights.append(f"Model F1-score: {metrics.get('f1', 0):.3f}")
            
            if metrics.get('auc', 0) > 0.8:
                insights.append("Model shows strong predictive power")
            elif metrics.get('auc', 0) < 0.6:
                insights.append("Model needs improvement - consider adding more features")
        
        # 阈值优化洞察
        if 'threshold_optimization' in self.analysis_results:
            thresh = self.analysis_results['threshold_optimization']
            insights.append(f"Optimal threshold: {thresh['optimal_threshold']:.3f} "
                          f"(F1: {thresh['optimal_f1']:.3f})")
        
        return insights


class SatisfactionVisualizer:
    """
    满意度可视化器
    """
    
    def __init__(self, predictor: SatisfactionPredictor):
        self.predictor = predictor
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        绘制特征重要性
        
        Args:
            top_n: 显示前N个特征
            save_path: 保存路径
        """
        if self.predictor.feature_importance is None:
            logger.warning("No feature importance available")
            return
        
        importance_df = self.predictor.feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'].values)
        plt.yticks(range(len(importance_df)), importance_df['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test: pd.Series, y_proba: np.ndarray, save_path: Optional[str] = None):
        """
        绘制ROC曲线
        
        Args:
            y_test: 真实标签
            y_proba: 预测概率
            save_path: 保存路径
        """
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_test: pd.Series, y_pred: np.ndarray, save_path: Optional[str] = None):
        """
        绘制混淆矩阵
        
        Args:
            y_test: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(self, y_test: pd.Series, y_proba: np.ndarray, save_path: Optional[str] = None):
        """
        绘制预测概率分布
        
        Args:
            y_test: 真实标签
            y_proba: 预测概率
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))
        
        # 分别绘制正负样本的概率分布
        plt.hist(y_proba[y_test == 0], bins=30, alpha=0.5, label='Actual Negative', color='red')
        plt.hist(y_proba[y_test == 1], bins=30, alpha=0.5, label='Actual Positive', color='green')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_test: pd.Series, y_proba: np.ndarray, save_path: Optional[str] = None):
        """
        绘制精确率-召回率曲线
        
        Args:
            y_test: 真实标签
            y_proba: 预测概率
            save_path: 保存路径
        """
        from sklearn.metrics import precision_recall_curve
        
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, marker='.', label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    n = 10000
    
    # 特征
    df = pd.DataFrame({
        'user_id': range(n),
        'age': np.random.normal(35, 10, n),
        'investment_exp': np.random.exponential(5, n),
        'risk_score': np.random.uniform(0, 10, n),
        'query_frequency': np.random.poisson(10, n),
        'avg_rating_history': np.random.normal(3.5, 1, n),
        'nps_score_history': np.random.normal(0, 5, n),
        'query_length': np.random.normal(50, 20, n),
        'is_weekend': np.random.binomial(1, 0.3, n),
        'is_trading_hour': np.random.binomial(1, 0.6, n)
    })
    
    # 生成满意度标签（基于特征的逻辑）
    logit = (0.1 * df['age'] + 
             0.2 * df['investment_exp'] + 
             -0.1 * df['risk_score'] + 
             0.15 * df['query_frequency'] + 
             0.3 * df['avg_rating_history'] + 
             -0.2 * (df['nps_score_history'] < -5) +
             np.random.normal(0, 1, n))
    
    prob = 1 / (1 + np.exp(-logit))
    df['satisfied'] = (prob > 0.5).astype(int)
    
    # 特征列
    feature_cols = ['age', 'investment_exp', 'risk_score', 'query_frequency',
                   'avg_rating_history', 'nps_score_history', 'query_length',
                   'is_weekend', 'is_trading_hour']
    
    # 初始化预测器
    predictor = SatisfactionPredictor(model_type='xgboost')
    
    # 准备数据
    data = predictor.prepare_features(
        df=df,
        feature_cols=feature_cols,
        target_col='satisfied',
        test_size=0.2
    )
    
    # 训练模型
    predictor.train(data['X_train'], data['y_train'])
    
    # 评估模型
    metrics = predictor.evaluate(data['X_test'], data['y_test'])
    print("\n📊 Model Performance:")
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            print(f"  {metric}: {value:.4f}")
    
    # 特征重要性
    importance = predictor.get_feature_importance()
    print("\n🔝 Top 5 Feature Importance:")
    print(importance.head())
    
    # 初始化分析器
    analyzer = SatisfactionAnalyzer(predictor)
    
    # 添加预测列
    df['predicted'] = predictor.predict(df[feature_cols])
    df['predicted_proba'] = predictor.predict_proba(df[feature_cols])[:, 1]
    
    # 群体分析
    segments = analyzer.analyze_by_segments(
        df=df,
        segment_cols=['is_weekend', 'is_trading_hour'],
        target_col='satisfied',
        pred_col='predicted'
    )
    print("\n👥 Segment Analysis:")
    print(segments)
    
    # 阈值优化
    threshold = analyzer.find_optimal_threshold(
        data['y_test'],
        predictor.predict_proba(data['X_test'])[:, 1]
    )
    print(f"\n⚡ Optimal threshold: {threshold['optimal_threshold']:.3f}")
    
    # 生成洞察
    insights = analyzer.generate_insights()
    print("\n💡 Business Insights:")
    for insight in insights:
        print(f"  • {insight}")
    
    # 可视化
    visualizer = SatisfactionVisualizer(predictor)
    visualizer.plot_feature_importance(top_n=5)
    visualizer.plot_roc_curve(data['y_test'], predictor.predict_proba(data['X_test'])[:, 1])
    visualizer.plot_confusion_matrix(data['y_test'], predictor.predict(data['X_test']))
    
    # 保存模型
    predictor.save_model('models/satisfaction_model.pkl')
    
    # 加载模型
    new_predictor = SatisfactionPredictor()
    new_predictor.load_model('models/satisfaction_model.pkl')