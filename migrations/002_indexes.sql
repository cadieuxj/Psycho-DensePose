-- Indexes and optimizations for QuestDB

-- Note: QuestDB automatically indexes SYMBOL columns and the designated timestamp column
-- These are conceptual optimizations for query patterns

-- Common query patterns:
-- 1. Get all data for a subject within a time range
-- 2. Get latest predictions for active sessions
-- 3. Join trajectory with ocean predictions
-- 4. Aggregate metrics by time windows

-- QuestDB-specific optimizations:

-- Sample rate downsample for visualization
-- (QuestDB supports SAMPLE BY in queries, no index needed)

-- Deduplication configuration (optional, for high-frequency writes)
-- ALTER TABLE csi_raw DEDUP UPSERT KEYS(ts, antenna_id, subcarrier_index);

-- Retention policies (auto-delete old data)
-- Raw CSI: Keep only 7 days (high volume)
-- ALTER TABLE csi_raw DROP PARTITION LIST '2024-01-01', '2024-01-02'; -- manual for now

-- Movement metrics: Keep 30 days
-- OCEAN predictions: Keep 90 days (for analysis)
-- Sales intelligence: Keep 1 year (for training)

-- Common aggregation queries (these don't need indexes, but are optimized in QuestDB):

-- Latest OCEAN score per subject:
-- SELECT * FROM ocean_predictions
-- LATEST ON ts PARTITION BY subject_id;

-- Average hesitation by hour:
-- SELECT ts, avg(hesitation_score) as avg_hesitation
-- FROM movement_metrics
-- WHERE subject_id = 'xyz'
-- SAMPLE BY 1h;

-- Subject trajectory with metrics:
-- SELECT t.ts, t.position_x, t.position_y, m.hesitation_score
-- FROM trajectory_points t
-- ASOF JOIN movement_metrics m ON(subject_id)
-- WHERE t.subject_id = 'xyz';
