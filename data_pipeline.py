"""
DeepSeek金融垂域满意度提升项目 - 数据采集与处理管道
"""

import os
import pandas as pd
import numpy as np
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from sqlalchemy import create_engine, text
import tushare as ts
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinancialDataCollector:
    """
    金融数据采集器
    支持Tushare Pro、DeepSeek API等多源数据接入
    """
    
    def __init__(self, config_path: str = '../config/config.yaml'):
        """
        初始化数据采集器
        
        Args:
            config_path: 配置文件路径
        """
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.tushare_token = config['tushare']['token']
        self.deepseek_api_key = config['deepseek']['api_key']
        self.deepseek_api_base = config['deepseek'].get('api_base', 'https://api.deepseek.com')
        
        # 初始化Tushare
        ts.set_token(self.tushare_token)
        self.pro = ts.pro_api()
        
        # 数据库连接
        self.db_engine = create_engine(config['database']['url'])
        
        logger.info("FinancialDataCollector initialized")
    
    def get_stock_basic(self) -> pd.DataFrame:
        """
        获取股票基础信息
        
        Returns:
            DataFrame包含股票代码、名称、行业、上市日期等
        """
        try:
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date,is_hs'
            )
            
            # 转换字段名
            df = df.rename(columns={
                'ts_code': 'stock_code',
                'symbol': 'symbol',
                'name': 'stock_name',
                'area': 'area',
                'industry': 'industry',
                'market': 'market',
                'list_date': 'listing_date',
                'is_hs': 'is_hs'
            })
            
            # 添加市值字段（需要后续填充）
            df['market_cap'] = None
            
            logger.info(f"Retrieved {len(df)} stocks basic info")
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock basic info: {e}")
            return pd.DataFrame()
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        获取股票日线数据
        
        Args:
            stock_code: 股票代码（如 600519.SH）
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            DataFrame包含日线行情数据
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            df = self.pro.daily(
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                fields='trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
            )
            
            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            
            logger.info(f"Retrieved {len(df)} daily records for {stock_code}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting daily data for {stock_code}: {e}")
            return pd.DataFrame()
    
    def get_financial_data(self, stock_code: str, report_dates: List[str] = None) -> pd.DataFrame:
        """
        获取财务指标数据
        
        Args:
            stock_code: 股票代码
            report_dates: 报告期列表
            
        Returns:
            DataFrame包含财务数据
        """
        try:
            # 获取利润表数据
            income = self.pro.income_vip(
                ts_code=stock_code,
                fields='ts_code,end_date,revenue,net_profit,profit_deducted,operate_profit'
            )
            
            # 获取资产负债表数据
            balance = self.pro.balancesheet_vip(
                ts_code=stock_code,
                fields='ts_code,end_date,total_assets,total_liab,total_equity'
            )
            
            # 获取财务指标
            indicators = self.pro.fina_indicator_vip(
                ts_code=stock_code,
                fields='ts_code,end_date,roe,eps,pe,pe_ttm,pb,dividend_yield'
            )
            
            # 合并数据
            if not income.empty and not indicators.empty:
                # 转换日期格式
                for df in [income, balance, indicators]:
                    if not df.empty and 'end_date' in df.columns:
                        df['end_date'] = pd.to_datetime(df['end_date'])
                
                # 合并数据
                result = income.merge(indicators, on=['ts_code', 'end_date'], how='outer')
                if not balance.empty:
                    result = result.merge(balance[['ts_code', 'end_date', 'total_assets', 'total_liab', 'total_equity']], 
                                         on=['ts_code', 'end_date'], how='outer')
                
                result = result.rename(columns={'ts_code': 'stock_code'})
                logger.info(f"Retrieved financial data for {stock_code}, {len(result)} periods")
                return result
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting financial data for {stock_code}: {e}")
            return pd.DataFrame()
    
    def get_industry_classification(self) -> pd.DataFrame:
        """
        获取行业分类数据
        
        Returns:
            DataFrame包含股票行业分类
        """
        try:
            # 获取申万行业分类
            df = self.pro.industry_vip(
                src='sw2021',
                fields='ts_code,industry_name,industry_code,in_date,out_date,is_main'
            )
            
            # 筛选主要的行业分类（未过期且是主业）
            df = df[df['out_date'].isna() | (df['out_date'] == '')]
            df = df[df['is_main'] == 'Y']
            
            df = df.rename(columns={'ts_code': 'stock_code', 'industry_name': 'industry'})
            
            logger.info(f"Retrieved industry classification for {len(df)} stocks")
            return df[['stock_code', 'industry']]
            
        except Exception as e:
            logger.error(f"Error getting industry classification: {e}")
            return pd.DataFrame()
    
    def collect_user_queries(self, days: int = 30) -> pd.DataFrame:
        """
        模拟/采集用户查询数据（生产环境应从数据库读取）
        
        Args:
            days: 回溯天数
            
        Returns:
            DataFrame包含用户查询记录
        """
        # 注意：生产环境中，这里应该是从数据库或日志系统读取真实数据
        # 此处为演示目的生成模拟数据
        
        np.random.seed(42)
        
        # 获取股票列表
        stocks = self.get_stock_basic()
        if stocks.empty:
            logger.warning("No stocks data available, using mock stock list")
            stocks = pd.DataFrame({
                'stock_code': ['600519.SH', '000858.SZ', '601318.SH', '600036.SH', '000002.SZ'],
                'stock_name': ['贵州茅台', '五粮液', '中国平安', '招商银行', '万科A'],
                'industry': ['食品饮料', '食品饮料', '保险', '银行', '房地产'],
                'market_cap': [20000, 6000, 8000, 9000, 1500]
            })
        
        # 生成查询记录
        query_types = ['股价查询', '基本面分析', '技术分析', '行业对比', '财报解读', '投资建议']
        start_date = datetime.now() - timedelta(days=days)
        
        records = []
        for i in range(1000):  # 生成1000条模拟记录
            query_date = start_date + timedelta(
                days=np.random.randint(0, days),
                hours=np.random.randint(0, 23),
                minutes=np.random.randint(0, 59)
            )
            
            # 随机选择1-3只股票
            n_stocks = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            selected_stocks = stocks.sample(n_stocks)
            
            records.append({
                'query_id': i + 10000,
                'user_id': f'user_{np.random.randint(1, 201)}',
                'query_time': query_date,
                'query_type': np.random.choice(query_types),
                'stock_codes': json.dumps(selected_stocks['stock_code'].tolist()),
                'experiment_group': np.random.choice(['control', 'treatment_rag', 'treatment_prompt'], 
                                                     p=[0.4, 0.3, 0.3])
            })
        
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} mock user queries")
        return df
    
    def save_to_database(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """
        保存数据到数据库
        
        Args:
            df: 要保存的DataFrame
            table_name: 目标表名
            if_exists: 存在时处理方式 ('fail', 'replace', 'append')
        """
        try:
            df.to_sql(
                name=table_name,
                con=self.db_engine,
                if_exists=if_exists,
                index=False,
                method='multi'
            )
            logger.info(f"Saved {len(df)} records to {table_name}")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def load_from_database(self, query: str) -> pd.DataFrame:
        """
        从数据库加载数据
        
        Args:
            query: SQL查询语句
            
        Returns:
            DataFrame查询结果
        """
        try:
            df = pd.read_sql(query, self.db_engine)
            logger.info(f"Loaded {len(df)} records from database")
            return df
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            return pd.DataFrame()


class DataPreprocessor:
    """
    数据预处理器：清洗、转换、特征工程
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗股票数据
        
        Args:
            df: 原始DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        df_clean = df.copy()
        
        # 处理缺失值
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # 用前向填充+后向填充处理时间序列数据
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        # 删除全为空的列
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # 去重
        df_clean = df_clean.drop_duplicates()
        
        return df_clean
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            增加技术指标的DataFrame
        """
        df_tech = df.copy()
        
        # 确保数据按日期排序
        if 'trade_date' in df_tech.columns:
            df_tech = df_tech.sort_values('trade_date')
        
        # 简单移动平均线
        for window in [5, 10, 20, 30, 60]:
            df_tech[f'ma_{window}'] = df_tech['close'].rolling(window=window).mean()
        
        # 指数移动平均线
        for window in [12, 26]:
            df_tech[f'ema_{window}'] = df_tech['close'].ewm(span=window, adjust=False).mean()
        
        # MACD
        df_tech['ema_12'] = df_tech['close'].ewm(span=12, adjust=False).mean()
        df_tech['ema_26'] = df_tech['close'].ewm(span=26, adjust=False).mean()
        df_tech['macd'] = df_tech['ema_12'] - df_tech['ema_26']
        df_tech['macd_signal'] = df_tech['macd'].ewm(span=9, adjust=False).mean()
        df_tech['macd_hist'] = df_tech['macd'] - df_tech['macd_signal']
        
        # RSI
        delta = df_tech['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_tech['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df_tech['bb_middle'] = df_tech['close'].rolling(window=20).mean