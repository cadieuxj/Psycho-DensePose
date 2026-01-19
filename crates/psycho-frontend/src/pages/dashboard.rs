//! Sales dashboard with live intelligence display.

use leptos::*;
use psycho_core::{OceanScores, SalesPersona};

#[component]
pub fn Dashboard() -> impl IntoView {
    // Mock data - would come from WebTransport in production
    let (active_subjects, _) = create_signal(3);
    let (current_session, _) = create_signal(Some(MockSession::default()));

    view! {
        <div class="dashboard">
            <header class="dashboard-header">
                <h1>"Sales Intelligence Dashboard"</h1>
                <div class="status-badge">
                    <span class="indicator active"></span>
                    {move || format!("{} Active Subjects", active_subjects.get())}
                </div>
            </header>

            <div class="dashboard-grid">
                // Current session intelligence
                {move || current_session.get().map(|session| view! {
                    <SessionCard session=session/>
                })}

                // OCEAN visualization
                <OceanRadarChart scores=OceanScores::new(0.6, 0.7, 0.5, 0.6, 0.4)/>

                // Live trajectory feed
                <TrajectoryFeed/>

                // Conversation hooks
                <ConversationHooks/>
            </div>
        </div>
    }
}

#[component]
fn SessionCard(session: MockSession) -> impl IntoView {
    view! {
        <div class="card session-card">
            <h2>"Current Session"</h2>
            <div class="session-info">
                <div class="info-row">
                    <span class="label">"Persona:"</span>
                    <span class="value persona">{format!("{:?}", session.persona)}</span>
                </div>
                <div class="info-row">
                    <span class="label">"Time Spent:"</span>
                    <span class="value">{format!("{} min", session.time_spent_mins)}</span>
                </div>
                <div class="info-row">
                    <span class="label">"Confidence:"</span>
                    <span class="value">{format!("{:.0}%", session.confidence * 100.0)}</span>
                </div>
            </div>
            <div class="behavioral-summary">
                <h3>"Behavioral Summary"</h3>
                <p>{&session.summary}</p>
            </div>
        </div>
    }
}

#[component]
fn OceanRadarChart(scores: OceanScores) -> impl IntoView {
    view! {
        <div class="card ocean-chart">
            <h2>"OCEAN Profile"</h2>
            <div class="ocean-bars">
                <OceanBar label="Openness" value=scores.openness/>
                <OceanBar label="Conscientiousness" value=scores.conscientiousness/>
                <OceanBar label="Extraversion" value=scores.extraversion/>
                <OceanBar label="Agreeableness" value=scores.agreeableness/>
                <OceanBar label="Neuroticism" value=scores.neuroticism/>
            </div>
        </div>
    }
}

#[component]
fn OceanBar(label: &'static str, value: f64) -> impl IntoView {
    let percentage = (value * 100.0) as u32;
    view! {
        <div class="ocean-bar-row">
            <span class="bar-label">{label}</span>
            <div class="bar-container">
                <div class="bar-fill" style=format!("width: {}%", percentage)></div>
            </div>
            <span class="bar-value">{format!("{:.2}", value)}</span>
        </div>
    }
}

#[component]
fn TrajectoryFeed() -> impl IntoView {
    view! {
        <div class="card trajectory-feed">
            <h2>"Live Movement"</h2>
            <div class="trajectory-list">
                <div class="trajectory-item">
                    <span class="subject-id">"Subject #1"</span>
                    <span class="location">"Near SUV section"</span>
                    <span class="hesitation high">"High hesitation"</span>
                </div>
                <div class="trajectory-item">
                    <span class="subject-id">"Subject #2"</span>
                    <span class="location">"Sedan display"</span>
                    <span class="hesitation low">"Decisive movement"</span>
                </div>
            </div>
        </div>
    }
}

#[component]
fn ConversationHooks() -> impl IntoView {
    let hooks = vec![
        "I noticed you're spending time looking at our safety-rated models. What features matter most to you?",
        "Many families tell us they appreciate the advanced driver assistance systems. Have you had a chance to test those?",
        "You seem to be taking your time evaluating options - that's great. What questions can I answer for you?",
    ];

    view! {
        <div class="card conversation-hooks">
            <h2>"Recommended Opening Hooks"</h2>
            <div class="hooks-list">
                {hooks.into_iter().enumerate().map(|(i, hook)| view! {
                    <div class="hook-item">
                        <span class="hook-number">{i + 1}</span>
                        <p class="hook-text">{hook}</p>
                        <button class="copy-btn">"Copy"</button>
                    </div>
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}

// Mock data structure
#[derive(Clone)]
struct MockSession {
    persona: SalesPersona,
    time_spent_mins: u32,
    confidence: f64,
    summary: String,
}

impl Default for MockSession {
    fn default() -> Self {
        Self {
            persona: SalesPersona::Analyst,
            time_spent_mins: 8,
            confidence: 0.82,
            summary: "Customer exhibits methodical, analytical behavior with high conscientiousness. Prolonged observation time suggests thorough evaluation of safety features and reliability metrics.".to_string(),
        }
    }
}
