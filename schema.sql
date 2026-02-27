

## 📄 sql/schema.sql

```sql
-- =====================================================
-- DeepSeek金融垂域满意度提升项目 - 数据库表结构
-- =====================================================

-- 1. 用户信息表
CREATE TABLE users (
    user_id VARCHAR(64) PRIMARY KEY,
    registration_date DATE,
    user_type ENUM('免费', '付费', '企业') DEFAULT '免费',
    risk_profile ENUM('保守', '稳健', '进取') DEFAULT '稳健',
    investment_experience INT COMMENT '投资经验年限',
    last_active_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 股票基本信息表
CREATE TABLE stocks (
    stock_code VARCHAR(16) PRIMARY KEY,
    stock_name VARCHAR(64),
    market ENUM('SSE', 'SZSE', 'HKEX', 'NASDAQ', 'NYSE'),
    industry VARCHAR(64),
    listing_date DATE,
    market_cap DECIMAL(20,2) COMMENT '市值(亿元)',
    is_index_component BOOLEAN COMMENT '是否指数成分股',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_industry (industry),
    INDEX idx_market_cap (market_cap)
);

-- 3. 用户查询日志表
CREATE TABLE user_queries (
    query_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64),
    session_id VARCHAR(128),
    query_text TEXT,
    query_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query_type ENUM('股价查询', '基本面分析', '技术分析', '行业对比', '财报解读', '投资建议'),
    stock_codes JSON COMMENT '查询中涉及的股票代码列表',
    intent_score FLOAT COMMENT '意图识别置信度',
    experiment_group VARCHAR(32) COMMENT 'AB实验分组',
    INDEX idx_user_id (user_id),
    INDEX idx_query_time (query_time),
    INDEX idx_experiment_group (experiment_group),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 4. 模型回答日志表
CREATE TABLE model_responses (
    response_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    query_id BIGINT,
    response_text LONGTEXT,
    response_time_ms INT COMMENT '响应耗时(毫秒)',
    model_version VARCHAR(32),
    prompt_template VARCHAR(128),
    retrieval_method ENUM('无检索', '基础RAG', '增强RAG', '结构化Prompt'),
    tokens_used INT,
    response_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_query_id (query_id),
    FOREIGN KEY (query_id) REFERENCES user_queries(query_id)
);

-- 5. 用户反馈表
CREATE TABLE user_feedback (
    feedback_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    query_id BIGINT,
    user_id VARCHAR(64),
    rating TINYINT COMMENT '1-5星评分',
    nps_score TINYINT COMMENT '0-10分',
    feedback_type ENUM('点赞', '点踩', '举报', '详细反馈'),
    feedback_reason JSON COMMENT '反馈原因多选',
    feedback_text TEXT COMMENT '用户评论文本',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_query_id (query_id),
    INDEX idx_user_id (user_id),
    FOREIGN KEY (query_id) REFERENCES user_queries(query_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 6. 回答质量人工评估表
CREATE TABLE quality_evaluation (
    eval_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    response_id BIGINT,
    evaluator VARCHAR(64),
    factual_accuracy TINYINT COMMENT '事实准确性0-10分',
    logical_completeness TINYINT COMMENT '逻辑完整性0-10分',
    timeliness TINYINT COMMENT '时效性0-10分',
    depth_score TINYINT COMMENT '分析深度0-10分',
    has_hallucination BOOLEAN COMMENT '是否包含幻觉',
    hallucination_detail TEXT COMMENT '幻觉详情',
    overall_score TINYINT COMMENT '综合评分',
    eval_date DATE,
    INDEX idx_response_id (response_id),
    FOREIGN KEY (response_id) REFERENCES model_responses(response_id)
);

-- 7. AB实验配置表
CREATE TABLE ab_experiments (
    experiment_id VARCHAR(64) PRIMARY KEY,
    experiment_name VARCHAR(128),
    description TEXT,
    start_date DATE,
    end_date DATE,
    status ENUM('设计', '运行', '暂停', '结束'),
    traffic_percentage INT COMMENT '实验流量占比%',
    control_group_name VARCHAR(32),
    treatment_groups JSON COMMENT '实验组配置',
    target_metrics JSON COMMENT '核心指标列表',
    created_by VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. 用户实验分组表
CREATE TABLE user_experiment_assignments (
    assignment_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64),
    experiment_id VARCHAR(64),
    group_name VARCHAR(32),
    assignment_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE KEY uk_user_experiment (user_id, experiment_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (experiment_id) REFERENCES ab_experiments(experiment_id)
);

-- 9. 股票财务数据表
CREATE TABLE financial_data (
    financial_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    stock_code VARCHAR(16),
    report_date DATE,
    revenue DECIMAL(20,2) COMMENT '营业收入(亿元)',
    net_profit DECIMAL(20,2) COMMENT '净利润(亿元)',
    gross_margin FLOAT COMMENT '毛利率%',
    net_margin FLOAT COMMENT '净利率%',
    roe FLOAT COMMENT '净资产收益率%',
    eps DECIMAL(10,3) COMMENT '每股收益',
    pe_ttm FLOAT COMMENT '市盈率TTM',
    pb FLOAT COMMENT '市净率',
    dividend_yield FLOAT COMMENT '股息率%',
    data_source VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_stock_code (stock_code),
    INDEX idx_report_date (report_date)
);

-- 10. 实验效果指标表
CREATE TABLE experiment_metrics (
    metric_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(64),
    group_name VARCHAR(32),
    date DATE,
    metric_name VARCHAR(64),
    metric_value FLOAT,
    sample_size INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_experiment_date (experiment_id, date)
);

-- =====================================================
-- 视图创建
-- =====================================================

-- 满意度分析视图
CREATE VIEW v_satisfaction_analysis AS
SELECT 
    uq.query_date,
    uq.experiment_group,
    s.industry,
    CASE 
        WHEN s.market_cap > 1000 THEN '大盘股'
        WHEN s.market_cap BETWEEN 100 AND 1000 THEN '中盘股'
        WHEN s.market_cap BETWEEN 10 AND 100 THEN '小盘股'
        ELSE '微盘股'
    END AS stock_size,
    COUNT(DISTINCT uq.query_id) AS query_count,
    AVG(uf.rating) AS avg_rating,
    AVG(uf.nps_score) AS avg_nps,
    SUM(CASE WHEN uf.rating >= 4 THEN 1 ELSE 0 END) / COUNT(*) AS satisfaction_rate
FROM user_queries uq
JOIN user_feedback uf ON uq.query_id = uf.query_id
JOIN stocks s ON JSON_CONTAINS(uq.stock_codes, CONCAT('"', s.stock_code, '"'))
GROUP BY uq.query_date, uq.experiment_group, s.industry, stock_size;