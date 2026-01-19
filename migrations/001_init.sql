-- QuestDB Schema for Psycho-DensePose System
-- Time-series optimized for high-throughput CSI ingestion

-- Raw CSI data table (high-frequency writes)
CREATE TABLE IF NOT EXISTS csi_raw (
    ts TIMESTAMP,
    antenna_id SYMBOL,
    subcarrier_index INT,
    amplitude DOUBLE,
    phase DOUBLE,
    rssi INT,
    noise_floor INT,
    sequence_number INT,
    tx_mac_hash LONG
) TIMESTAMP(ts) PARTITION BY DAY;

-- Doppler velocity estimates
CREATE TABLE IF NOT EXISTS doppler_estimates (
    ts TIMESTAMP,
    antenna_id SYMBOL,
    subject_id SYMBOL,
    velocity DOUBLE,
    doppler_frequency DOUBLE,
    power DOUBLE,
    confidence DOUBLE
) TIMESTAMP(ts) PARTITION BY DAY;

-- DensePose keypoint detections
CREATE TABLE IF NOT EXISTS keypoint_detections (
    ts TIMESTAMP,
    subject_id SYMBOL,
    keypoint_type SYMBOL,
    position_x DOUBLE,
    position_y DOUBLE,
    position_z DOUBLE,
    confidence DOUBLE
) TIMESTAMP(ts) PARTITION BY DAY;

-- Trajectory points (subject movement)
CREATE TABLE IF NOT EXISTS trajectory_points (
    ts TIMESTAMP,
    subject_id SYMBOL,
    position_x DOUBLE,
    position_y DOUBLE,
    position_z DOUBLE,
    velocity_x DOUBLE,
    velocity_y DOUBLE,
    velocity_z DOUBLE,
    acceleration_x DOUBLE,
    acceleration_y DOUBLE,
    acceleration_z DOUBLE
) TIMESTAMP(ts) PARTITION BY DAY;

-- Movement metrics (aggregated features)
CREATE TABLE IF NOT EXISTS movement_metrics (
    ts TIMESTAMP,
    subject_id SYMBOL,
    window_duration_secs DOUBLE,

    -- Kinematic features
    path_efficiency_ratio DOUBLE,
    mean_velocity DOUBLE,
    max_velocity DOUBLE,
    velocity_variance DOUBLE,
    mean_acceleration DOUBLE,
    rms_jerk DOUBLE,
    max_jerk DOUBLE,
    direction_changes INT,

    -- LMA effort features
    space_directness DOUBLE,
    time_suddenness DOUBLE,
    weight_strength DOUBLE,
    flow_boundness DOUBLE,

    -- Behavioral features
    hesitation_score DOUBLE,
    kinesphere_volume DOUBLE,
    path_entropy DOUBLE,
    fidgeting_score DOUBLE
) TIMESTAMP(ts) PARTITION BY HOUR;

-- OCEAN personality predictions
CREATE TABLE IF NOT EXISTS ocean_predictions (
    ts TIMESTAMP,
    subject_id SYMBOL,
    session_id SYMBOL,

    -- OCEAN scores
    openness DOUBLE,
    conscientiousness DOUBLE,
    extraversion DOUBLE,
    agreeableness DOUBLE,
    neuroticism DOUBLE,

    -- Metadata
    confidence SYMBOL,
    observation_count INT,
    observation_duration_secs DOUBLE
) TIMESTAMP(ts) PARTITION BY HOUR;

-- Kiosk interaction events
CREATE TABLE IF NOT EXISTS kiosk_events (
    ts TIMESTAMP,
    session_id SYMBOL,
    kiosk_id SYMBOL,
    event_type SYMBOL,
    position_x DOUBLE,
    position_y DOUBLE,
    position_z DOUBLE
) TIMESTAMP(ts) PARTITION BY DAY;

-- Handshake matches (WiFi trajectory → Kiosk session)
CREATE TABLE IF NOT EXISTS handshake_matches (
    ts TIMESTAMP,
    subject_id SYMBOL,
    session_id SYMBOL,
    confidence DOUBLE,
    distance_score DOUBLE,
    orientation_score DOUBLE,
    velocity_score DOUBLE,
    timing_score DOUBLE,
    deceleration_score DOUBLE
) TIMESTAMP(ts) PARTITION BY DAY;

-- Sales intelligence results
CREATE TABLE IF NOT EXISTS sales_intelligence (
    ts TIMESTAMP,
    subject_id SYMBOL,
    session_id SYMBOL,

    -- Agent outputs (stored as JSON-like text for flexibility)
    sales_approach SYMBOL,
    emotional_state STRING,
    key_themes STRING,
    recommended_features STRING,
    communication_style STRING,

    -- Hooks count
    hooks_generated INT
) TIMESTAMP(ts) PARTITION BY HOUR;

-- Questionnaire responses
CREATE TABLE IF NOT EXISTS questionnaire_responses (
    ts TIMESTAMP,
    session_id SYMBOL,
    vehicle_type STRING,
    budget_range STRING,
    primary_use STRING,
    priorities STRING,
    has_trade_in BOOLEAN,
    financing_preference STRING
) TIMESTAMP(ts) PARTITION BY DAY;

-- System metrics for monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    ts TIMESTAMP,
    metric_name SYMBOL,
    metric_value DOUBLE,
    metric_unit SYMBOL,
    component SYMBOL
) TIMESTAMP(ts) PARTITION BY HOUR;
