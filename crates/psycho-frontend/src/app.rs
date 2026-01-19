//! Main application component with routing.

use leptos::*;
use leptos_meta::*;
use leptos_router::*;

use crate::pages::{AdminPanel, Dashboard, Visualizer};

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/psycho-frontend.css"/>
        <Title text="Psycho-DensePose"/>
        <Meta name="description" content="WiFi-based DensePose and Psychometric Profiling"/>

        <Router>
            <nav class="navbar">
                <div class="nav-brand">
                    <h1>"Psycho-DensePose"</h1>
                </div>
                <div class="nav-links">
                    <A href="/">"Dashboard"</A>
                    <A href="/visualizer">"3D Visualizer"</A>
                    <A href="/admin">"Admin"</A>
                </div>
            </nav>

            <main class="container">
                <Routes>
                    <Route path="/" view=Dashboard/>
                    <Route path="/visualizer" view=Visualizer/>
                    <Route path="/admin" view=AdminPanel/>
                    <Route path="/*any" view=NotFound/>
                </Routes>
            </main>
        </Router>
    }
}

#[component]
fn NotFound() -> impl IntoView {
    view! {
        <div class="not-found">
            <h1>"404"</h1>
            <p>"Page not found"</p>
            <A href="/">"Return to Dashboard"</A>
        </div>
    }
}
