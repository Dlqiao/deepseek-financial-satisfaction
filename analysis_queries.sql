-- =====================================================
-- DeepSeek金融垂域满意度提升项目 - 分析查询脚本
-- =====================================================

-- 1. 总体满意度趋势分析
SELECT 
    DATE(query_time) as query_date,
    COUNT(*) as total_queries,
    AVG(rating) as avg_rating,
    AVG(nps_score) as avg_nps,
    SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as satisfaction_rate
FROM user_queries uq
JOIN user_feedback uf ON uq.query_id = uf.query_id
GROUP BY DATE(query_time)
ORDER BY query_date;

-- 2. 不同市值股票满意度对比（验证"吃不饱"现象）
SELECT 
    CASE 
        WHEN s.market_cap > 1000 THEN '大盘股 (>1000亿)'
        WHEN s.market_cap BETWEEN 100 AND 1000 THEN '中盘股 (100-1000亿)'
        WHEN s.market_cap BETWEEN 10 AND 100 THEN '小盘股 (10-100亿)'
        ELSE '微盘股 (<10亿)'
    END as stock_category,
    COUNT(DISTINCT uq.query_id) as query_count,
    AVG(uf.rating) as avg_rating,
    AVG(uf.nps_score) as avg_nps,
    AVG(qe.depth_score) as avg_depth_score,
    SUM(CASE WHEN qe.has_hallucination THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as hallucination_rate
FROM user_queries uq
JOIN user_feedback uf ON uq.query_id = uf.query_id
JOIN model_responses mr ON uq.query_id = mr.query_id
LEFT JOIN quality_evaluation qe ON mr.response_id = qe.response_id
JOIN stocks s ON JSON_CONTAINS(uq.stock_codes, CONCAT('"', s.stock_code, '"'))
GROUP BY stock_category
ORDER BY stock_category;

-- 3. AB实验效果对比
SELECT 
    uq.experiment_group,
    mr.retrieval_method,
    COUNT(DISTINCT uq.query_id) as query_count,
    AVG(uf.rating) as avg_rating,
    AVG(uf.nps_score) as avg_nps,
    AVG(mr.response_time_ms) as avg_response_time,
    SUM(CASE WHEN uf.rating >= 4 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as satisfaction_rate,
    AVG(qe.factual_accuracy) as avg_factual_accuracy,
    AVG(qe.logical_completeness) as avg_logical_completeness
FROM user_queries uq
JOIN model_responses mr ON uq.query_id = mr.query_id
JOIN user_feedback uf ON uq.query_id = uf.query_id
LEFT JOIN quality_evaluation qe ON mr.response_id = qe.response_id
WHERE uq.experiment_group IS NOT NULL
GROUP BY uq.experiment_group, mr.retrieval_method
ORDER BY satisfaction_rate DESC;

-- 4. 因果推断准备数据：PSM所需的用户特征表
SELECT 
    u.user_id,
    u.user_type,
    u.risk_profile,
    u.investment_experience,
    COUNT(DISTINCT uq.query_id) as total_queries_30d,
    COUNT(DISTINCT CASE WHEN uq.query_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) THEN uq.query_id END) as queries_7d,
    AVG(uf.rating) as historical_avg_rating,
    ue.group_name as experiment_group,
    CASE WHEN u.last_active_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) THEN 1 ELSE 0 END as is_active_next_week
FROM users u
LEFT JOIN user_queries uq ON u.user_id = uq.user_id AND uq.query_time >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
LEFT JOIN user_feedback uf ON uq.query_id = uf.query_id
LEFT JOIN user_experiment_assignments ue ON u.user_id = ue.user_id AND ue.is_active = TRUE
GROUP BY u.user_id, u.user_type, u.risk_profile, u.investment_experience, ue.group_name;

-- 5. 实验组间差异显著性检验（t-test准备数据）
SELECT 
    group_name,
    COUNT(*) as sample_size,
    AVG(nps_score) as mean_nps,
    STDDEV(nps_score) as std_nps
FROM user_feedback uf
JOIN user_queries uq ON uf.query_id = uq.query_id
WHERE uq.experiment_group IN ('control', 'treatment_rag', 'treatment_prompt')
GROUP BY group_name;

-- 6. 长尾股票挖掘效果评估
SELECT 
    mr.retrieval_method,
    COUNT(DISTINCT uq.query_id) as total_queries,
    SUM(CASE WHEN s.market_cap < 100 THEN 1 ELSE 0 END) as small_cap_queries,
    SUM(CASE WHEN s.market_cap < 100 AND uf.rating >= 4 THEN 1 ELSE 0 END) * 100.0 / 
        NULLIF(SUM(CASE WHEN s.market_cap < 100 THEN 1 ELSE 0 END), 0) as small_cap_satisfaction,
    AVG(CASE WHEN s.market_cap < 100 THEN qe.depth_score END) as small_cap_depth
FROM user_queries uq
JOIN model_responses mr ON uq.query_id = mr.query_id
JOIN user_feedback uf ON uq.query_id = uf.query_id
LEFT JOIN quality_evaluation qe ON mr.response_id = qe.response_id
JOIN stocks s ON JSON_CONTAINS(uq.stock_codes, CONCAT('"', s.stock_code, '"'))
GROUP BY mr.retrieval_method;

-- 7. 用户留存分析
SELECT 
    ue.group_name,
    DATE(u.last_active_date) as active_date,
    COUNT(DISTINCT u.user_id) as active_users,
    COUNT(DISTINCT CASE WHEN u2.last_active_date > DATE_ADD(u.last_active_date, INTERVAL 7 DAY) 
                   THEN u.user_id END) * 100.0 / COUNT(DISTINCT u.user_id) as retention_rate_7d
FROM users u
JOIN user_experiment_assignments ue ON u.user_id = ue.user_id
LEFT JOIN users u2 ON u.user_id = u2.user_id 
    AND u2.last_active_date > DATE_ADD(u.last_active_date, INTERVAL 7 DAY)
WHERE ue.is_active = TRUE
GROUP BY ue.group_name, DATE(u.last_active_date)
ORDER BY active_date;