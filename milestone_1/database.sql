

-- 1. Projects Table 
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    startup_name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    business_model VARCHAR(100),
    target_market VARCHAR(255),
    budget NUMERIC(15, 2),
    project_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. SWOT Analysis Table
CREATE TABLE IF NOT EXISTS swot_analysis (
    swot_id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    strengths TEXT,
    weaknesses TEXT,
    opportunities TEXT,
    threats TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Risk Assessments Table
CREATE TABLE IF NOT EXISTS risk_assessments (
    risk_id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    risk_category VARCHAR(100),
    risk_score NUMERIC(5, 2),
    risk_description TEXT,
    priority_level VARCHAR(50)
);

-- 4. Success Predictions Table
CREATE TABLE IF NOT EXISTS success_predictions (
    prediction_id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    success_probability NUMERIC(5, 2),
    overall_risk_score NUMERIC(5, 2),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Recommendations Table
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    recommendation_text TEXT,
    risk_mitigation TEXT,
    priority VARCHAR(50),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Assessment Reports Table
CREATE TABLE IF NOT EXISTS assessment_reports (
    report_id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    report_path VARCHAR(500),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);