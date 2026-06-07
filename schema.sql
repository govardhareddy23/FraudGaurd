-- Credit Card Fraud Detection — PostgreSQL Schema
-- Run: psql -U postgres -d fraud_detection -f schema.sql

CREATE DATABASE IF NOT EXISTS fraud_detection;
\c fraud_detection;

CREATE TABLE IF NOT EXISTS transactions (
    id                 SERIAL PRIMARY KEY,
    timestamp          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    amount             NUMERIC(12, 2) NOT NULL,
    time_val           NUMERIC(12, 4) DEFAULT 0,

    -- PCA features
    v1  NUMERIC(10,6) DEFAULT 0; v2  NUMERIC(10,6) DEFAULT 0;
    v3  NUMERIC(10,6) DEFAULT 0; v4  NUMERIC(10,6) DEFAULT 0;
    v5  NUMERIC(10,6) DEFAULT 0; v6  NUMERIC(10,6) DEFAULT 0;
    v7  NUMERIC(10,6) DEFAULT 0; v8  NUMERIC(10,6) DEFAULT 0;
    v9  NUMERIC(10,6) DEFAULT 0; v10 NUMERIC(10,6) DEFAULT 0;
    v11 NUMERIC(10,6) DEFAULT 0; v12 NUMERIC(10,6) DEFAULT 0;
    v13 NUMERIC(10,6) DEFAULT 0; v14 NUMERIC(10,6) DEFAULT 0;
    v15 NUMERIC(10,6) DEFAULT 0; v16 NUMERIC(10,6) DEFAULT 0;
    v17 NUMERIC(10,6) DEFAULT 0; v18 NUMERIC(10,6) DEFAULT 0;
    v19 NUMERIC(10,6) DEFAULT 0; v20 NUMERIC(10,6) DEFAULT 0;
    v21 NUMERIC(10,6) DEFAULT 0; v22 NUMERIC(10,6) DEFAULT 0;
    v23 NUMERIC(10,6) DEFAULT 0; v24 NUMERIC(10,6) DEFAULT 0;
    v25 NUMERIC(10,6) DEFAULT 0; v26 NUMERIC(10,6) DEFAULT 0;
    v27 NUMERIC(10,6) DEFAULT 0; v28 NUMERIC(10,6) DEFAULT 0;

    -- Prediction output
    fraud_probability  NUMERIC(6,4) NOT NULL,
    is_fraud           BOOLEAN NOT NULL,
    risk_level         VARCHAR(10) NOT NULL CHECK (risk_level IN ('LOW','MEDIUM','HIGH')),
    alert_triggered    BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS alerts (
    id                SERIAL PRIMARY KEY,
    transaction_id    INTEGER NOT NULL REFERENCES transactions(id),
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fraud_probability NUMERIC(6,4) NOT NULL,
    amount            NUMERIC(12,2) NOT NULL,
    status            VARCHAR(20) NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open','reviewed','dismissed')),
    notes             TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_txn_timestamp   ON transactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_txn_is_fraud    ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_alert_status    ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alerts(timestamp DESC);

-- Helpful views
CREATE OR REPLACE VIEW fraud_summary AS
SELECT
    DATE_TRUNC('hour', timestamp) AS hour,
    COUNT(*) AS total,
    SUM(is_fraud::int) AS fraud_count,
    ROUND(AVG(fraud_probability) * 100, 2) AS avg_fraud_pct,
    ROUND(AVG(amount), 2) AS avg_amount
FROM transactions
GROUP BY 1
ORDER BY 1 DESC;
