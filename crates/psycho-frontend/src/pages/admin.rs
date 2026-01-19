//! Admin panel for system monitoring and privacy controls.

use leptos::*;

#[component]
pub fn AdminPanel() -> impl IntoView {
    let (system_status, _) = create_signal(SystemStatus::default());

    view! {
        <div class="admin-panel">
            <header class="admin-header">
                <h1>"System Administration"</h1>
            </header>

            <div class="admin-grid">
                // System health
                <SystemHealth status=system_status/>

                // Privacy controls
                <PrivacyControls/>

                // Data retention
                <DataRetention/>

                // Performance metrics
                <PerformanceMetrics/>
            </div>
        </div>
    }
}

#[component]
fn SystemHealth(status: ReadSignal<SystemStatus>) -> impl IntoView {
    view! {
        <div class="card system-health">
            <h2>"System Health"</h2>
            <div class="health-grid">
                <HealthIndicator
                    label="CSI Pipeline"
                    status=move || status.get().csi_pipeline
                />
                <HealthIndicator
                    label="DensePose Model"
                    status=move || status.get().densepose_model
                />
                <HealthIndicator
                    label="LMA Analysis"
                    status=move || status.get().lma_analysis
                />
                <HealthIndicator
                    label="Multi-Agent System"
                    status=move || status.get().agent_system
                />
                <HealthIndicator
                    label="QuestDB"
                    status=move || status.get().database
                />
                <HealthIndicator
                    label="WebTransport"
                    status=move || status.get().webtransport
                />
            </div>
        </div>
    }
}

#[component]
fn HealthIndicator<F>(label: &'static str, status: F) -> impl IntoView
where
    F: Fn() -> HealthStatus + 'static,
{
    view! {
        <div class="health-indicator">
            <span class="health-label">{label}</span>
            <span class={move || format!("health-status {}",
                match status() {
                    HealthStatus::Healthy => "healthy",
                    HealthStatus::Degraded => "degraded",
                    HealthStatus::Failed => "failed",
                }
            )}>
                {move || format!("{:?}", status())}
            </span>
        </div>
    }
}

#[component]
fn PrivacyControls() -> impl IntoView {
    view! {
        <div class="card privacy-controls">
            <h2>"Privacy Controls"</h2>
            <div class="control-section">
                <h3>"Data Processing"</h3>
                <div class="toggle-row">
                    <span>"Edge Processing Only"</span>
                    <input type="checkbox" checked/>
                </div>
                <div class="toggle-row">
                    <span>"Anonymous Tracking"</span>
                    <input type="checkbox" checked/>
                </div>
            </div>

            <div class="control-section">
                <h3>"Subject Rights"</h3>
                <button class="action-btn">"Export Subject Data"</button>
                <button class="action-btn danger">"Delete Subject Data"</button>
            </div>

            <div class="control-section">
                <h3>"Consent Management"</h3>
                <p class="info-text">
                    "Kiosk interaction implies consent for session tracking"
                </p>
                <button class="action-btn">"View Consent Log"</button>
            </div>
        </div>
    }
}

#[component]
fn DataRetention() -> impl IntoView {
    view! {
        <div class="card data-retention">
            <h2>"Data Retention"</h2>
            <div class="retention-table">
                <RetentionRow table="csi_raw" retention="7 days" size="2.3 GB"/>
                <RetentionRow table="trajectory_points" retention="30 days" size="450 MB"/>
                <RetentionRow table="ocean_predictions" retention="90 days" size="12 MB"/>
                <RetentionRow table="sales_intelligence" retention="1 year" size="85 MB"/>
            </div>
            <button class="action-btn">"Purge Expired Data"</button>
        </div>
    }
}

#[component]
fn RetentionRow(table: &'static str, retention: &'static str, size: &'static str) -> impl IntoView {
    view! {
        <div class="retention-row">
            <span class="table-name">{table}</span>
            <span class="retention-period">{retention}</span>
            <span class="data-size">{size}</span>
        </div>
    }
}

#[component]
fn PerformanceMetrics() -> impl IntoView {
    view! {
        <div class="card performance-metrics">
            <h2>"Performance"</h2>
            <div class="metrics-grid">
                <MetricCard label="CSI Throughput" value="1000 Hz" unit="packets/sec"/>
                <MetricCard label="DensePose Latency" value="45 ms" unit="per frame"/>
                <MetricCard label="OCEAN Inference" value="120 ms" unit="per subject"/>
                <MetricCard label="Agent Generation" value="2.5 s" unit="per session"/>
            </div>
        </div>
    }
}

#[component]
fn MetricCard(label: &'static str, value: &'static str, unit: &'static str) -> impl IntoView {
    view! {
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-unit">{unit}</div>
        </div>
    }
}

// Data structures
#[derive(Clone, Copy)]
struct SystemStatus {
    csi_pipeline: HealthStatus,
    densepose_model: HealthStatus,
    lma_analysis: HealthStatus,
    agent_system: HealthStatus,
    database: HealthStatus,
    webtransport: HealthStatus,
}

impl Default for SystemStatus {
    fn default() -> Self {
        Self {
            csi_pipeline: HealthStatus::Healthy,
            densepose_model: HealthStatus::Healthy,
            lma_analysis: HealthStatus::Healthy,
            agent_system: HealthStatus::Healthy,
            database: HealthStatus::Healthy,
            webtransport: HealthStatus::Degraded,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum HealthStatus {
    Healthy,
    Degraded,
    Failed,
}
