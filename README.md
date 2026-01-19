# Psycho-DensePose: Architecture of Invisible Insight

> WiFi-based DensePose Estimation and Consumer Psychometric Profiling for Car Dealerships

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technical Stack](#technical-stack)
- [Getting Started](#getting-started)
- [Phase Implementations](#phase-implementations)
- [Database Schema](#database-schema)
- [API Endpoints](#api-endpoints)
- [Privacy & GDPR](#privacy--gdpr)
- [Contributing](#contributing)

## 🎯 Overview

Psycho-DensePose is a production-grade system that uses WiFi signals (Channel State Information) to estimate human pose and extract psychological traits, enabling personalized customer engagement in automotive dealerships without cameras.

### Key Capabilities

- **WiFi Sensing**: Process CSI from Intel AX210 (WiFi 6E, 160MHz, 1992 subcarriers)
- **DensePose Estimation**: Neural translation of 1D RF → 2D UV body surface mapping
- **Psychometric Analysis**: Laban Movement Analysis → Big Five (OCEAN) personality traits
- **Sales Intelligence**: Multi-agent system generating personalized engagement strategies
- **Privacy-First**: Edge processing, anonymous tracking, GDPR compliant

### Use Case

Customer walks into dealership → WiFi tracks movement → DensePose estimates pose → LMA analyzes behavior → OCEAN predicts personality → Customer uses kiosk → Handshake links trajectory to session → Multi-agent system generates personalized sales strategy with opening conversation hooks.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PSYCHO-DENSEPOSE SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

                          ┌──────────────┐
                          │ WiFi 6E NIC  │
                          │ Intel AX210  │
                          │ ESP32-S3     │
                          └──────┬───────┘
                                 │ CSI Data (1992 subcarriers @ 1kHz)
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 1: SIGNAL ENGINE (psycho-csi)                           │
│ • CSI Sanitization (phase unwrap, SFO removal, Hampel filter) │
│ • Doppler Extraction (FFT → velocity estimation)              │
│ • Feature Computation (amplitude/phase statistics)            │
└────────────────────────┬───────────────────────────────────────┘
                         │ Sanitized CSI
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 2: DENSEPOSE TRANSLATION (psycho-densepose)             │
│ • Dual-Branch Encoder (amplitude + phase)                     │
│ • Transformer Attention (isolate human Doppler)               │
│ • ResNet-50 Backbone (feature extraction)                     │
│ • Keypoint Head (17 joints) + DensePose Head (24 parts + UV)  │
│ • Cross-Modal Distillation (Teacher: Camera, Student: WiFi)   │
└────────────────────────┬───────────────────────────────────────┘
                         │ Skeletal Pose + DensePose
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 3: PSYCHOMETRIC ANALYSIS (psycho-lma)                   │
│ • Laban Movement Analysis (Space, Time, Weight, Flow)         │
│ • Hesitation Metric (integral of velocity⁻¹ × angular rate)   │
│ • Kinesphere Analysis (personal space utilization)            │
│ • OCEAN Mapping (movement features → Big Five traits)         │
│ • Sales Persona Classification                                │
└────────────────────────┬───────────────────────────────────────┘
                         │ Psychometric Profile
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 4: PRIVACY & HANDSHAKE (psycho-privacy)                 │
│ • Spatio-Temporal Handshake (link WiFi → Kiosk session)       │
│ • Multi-Factor Matching (position, orientation, timing)       │
│ • JPDA (Joint Probabilistic Data Association)                 │
│ • Anonymous Tracking (ephemeral UUIDs, edge processing)       │
└────────────────────────┬───────────────────────────────────────┘
                         │ Session Linkage
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 5: SALES INTELLIGENCE (psycho-agents)                   │
│ Agent 1 (Interpreter): LMA + OCEAN → Behavioral Summary       │
│ Agent 2 (Strategist): Behavioral + Questionnaire → Strategy   │
│ Agent 3 (Copywriter): Strategy → 3 Opening Hooks (CoT)        │
└────────────────────────┬───────────────────────────────────────┘
                         │ Sales Intelligence
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ API LAYER (psycho-api)                                         │
│ • REST API (HTTP/2): Sessions, Questionnaires, Intelligence   │
│ • WebTransport (HTTP/3): Real-time CSI/Pose/Updates           │
│ • QuestDB: Time-series storage (7-day CSI, 90-day metrics)    │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ FRONTEND (Leptos + WebGPU)                                     │
│ • 3D Visualization (skeletal tracking, trajectory heatmaps)   │
│ • Sales Dashboard (live intelligence, conversation hooks)     │
│ • Admin Panel (system health, privacy controls)               │
└────────────────────────────────────────────────────────────────┘
```

## 💻 Technical Stack (2026 Standards)

### Back-end
- **Language**: Rust 2024 Edition
- **Framework**: Axum 0.8 (HTTP/2 + WebTransport)
- **WebTransport**: Quinn 0.11 (QUIC), wtransport 0.5
- **ML Framework**: Candle (pure Rust inference)
- **LLM**: Llama 3 (local inference via QLoRA)

### Database
- **Time-Series**: QuestDB (high-throughput CSI ingestion)
- **Query**: SQL with SAMPLE BY, ASOF JOIN for time-series

### Front-end
- **Framework**: Leptos 0.6 (Rust → WASM)
- **Graphics**: WebGPU (3D skeletal interpolation, compute shaders)
- **Real-time**: WebTransport for bi-directional streaming

### Hardware
- **WiFi NIC**: Intel AX210 (WiFi 6E, 6GHz, 160MHz bandwidth)
- **Middleware**: PicoScenes for CSI extraction
- **Mesh Nodes**: ESP32-S3 (distributed sensing)

## 🚀 Getting Started

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
rustup update

# QuestDB
docker pull questdb/questdb:latest

# PicoScenes (for Intel AX210)
# Follow: https://ps.zpj.io/download
```

### Installation

```bash
# Clone repository
git clone https://github.com/cadieuxj/Psycho-DensePose.git
cd Psycho-DensePose

# Build workspace
cargo build --release

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

### Configuration

```bash
# Copy example configuration
cp config.example.toml config.toml

# Edit configuration
nano config.toml

# Set environment variables
export PSYCHO_HTTP_BIND_ADDR=0.0.0.0:8080
export PSYCHO_QUESTDB_HOST=localhost
export PSYCHO_QUESTDB_ILP_PORT=9009
```

### Running

```bash
# Start QuestDB
docker run -p 9000:9000 -p 9009:9009 -p 8812:8812 questdb/questdb

# Run database migrations
psql -h localhost -p 8812 -d qdb -f migrations/001_init.sql

# Start API server
cargo run --release --bin psycho-api

# Start frontend dev server
cd crates/psycho-frontend
trunk serve --port 3000
```

## 📊 Phase Implementations

### Phase 1: Signal Engine ✅

**Location**: `crates/psycho-csi/`

**Features**:
- CSI packet structures for WiFi 6E (1992 subcarriers)
- Phase unwrapping (remove 2π discontinuities)
- Linear phase removal (SFO + PDD compensation)
- Hampel filter (outlier rejection, MAD-based)
- Doppler extraction (FFT → velocity, 64-point window)
- Advanced filtering (Butterworth, Kalman, Savitzky-Golay)

**Key Files**:
- `packet.rs` - CSI data structures
- `sanitizer.rs` - Phase unwrapping, SFO removal
- `doppler.rs` - Doppler shift extraction
- `filtering.rs` - Signal processing filters
- `pipeline.rs` - Complete processing pipeline

### Phase 2: DensePose Translation ✅

**Location**: `crates/psycho-densepose/`

**Architecture**:
```
Input: CSI [batch, timesteps, subcarriers]
    ↓
Dual-Branch Encoder (amplitude | phase)
    ↓
Transformer Layers (4× with 8 attention heads)
    ↓
ResNet-50 Backbone → Feature Pyramid Network
    ↓
ROI Align → [Keypoint Head | DensePose Head]
    ↓
Output: 17 joints + 24 body parts + UV coordinates
```

**Loss Function**:
```
L_total = λ₁·L_distill + λ₂·L_keypoint + λ₃·L_uv + λ₄·L_part

L_distill = KL(P_teacher || P_student) × T²
L_keypoint = MSE_weighted(pred_heatmaps, target_heatmaps)
L_uv = SmoothL1(pred_uv, target_uv)
L_part = CrossEntropy(pred_parts, target_parts)
```

### Phase 3: Psychometric Analysis ✅

**Location**: `crates/psycho-lma/`

**Laban Movement Analysis**:

| Effort Factor | Metric | Interpretation |
|--------------|--------|----------------|
| **Space** | Path Efficiency Ratio (PER) | Direct vs Indirect attention |
| **Time** | RMS Jerk (m/s³) | Sudden vs Sustained movement |
| **Weight** | Mean Acceleration (m/s²) | Strong vs Light impact |
| **Flow** | Velocity Consistency | Bound vs Free control |

**Hesitation Formula**:
```
H(t) = ∫_{t-Δt}^{t} [1/(v(τ) + ε)] · |dθ/dτ| dτ

where:
  v(τ) = velocity at time τ
  θ = direction angle
  ε = 0.01 (avoid division by zero)
  Δt = 1.0 sec integration window
```

**OCEAN Mapping**:

| Movement Pattern | OCEAN Trait |
|-----------------|-------------|
| High path entropy, exploratory | → High **Openness** |
| High PER, low jerk, deliberate | → High **Conscientiousness** |
| Large kinesphere, fast pace | → High **Extraversion** |
| Smooth flow, accommodating | → High **Agreeableness** |
| Bound flow, high jerk, hesitation | → High **Neuroticism** |

### Phase 4: Privacy & Handshake ✅

**Location**: `crates/psycho-privacy/`

**Spatio-Temporal Handshake**:

Match criteria (all must be satisfied):
1. **Distance**: Subject within 2m of kiosk
2. **Orientation**: Facing kiosk (±45°)
3. **Velocity**: Near-stopped (v < 0.2 m/s)
4. **Timing**: Within ±2 sec of click event
5. **Deceleration**: Slowing down (a < -0.5 m/s²)

**Confidence Score**:
```
Confidence = 0.25×distance_score
           + 0.20×orientation_score
           + 0.25×velocity_score
           + 0.15×timing_score
           + 0.15×deceleration_score
```

Match accepted if Confidence ≥ 0.6

### Phase 5: Multi-Agent Intelligence ✅

**Location**: `crates/psycho-agents/`

**Agent Pipeline**:

```
┌─────────────────────────────────────┐
│ Agent 1: Interpreter                │
│ Input: OCEAN + LMA + Hesitation     │
│ Output: Behavioral Summary          │
│ "Customer shows caution with high   │
│  attention to safety features."     │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ Agent 2: Strategist                 │
│ Input: Behavioral + Questionnaire   │
│ Output: Sales Strategy              │
│ "Approach: Consultative             │
│  Themes: Safety, Reliability        │
│  Style: Calm, reassuring tone"      │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ Agent 3: Copywriter (CoT)           │
│ Output: 3 Opening Hooks             │
│ 1. "I noticed you're interested..." │
│ 2. "Safety is so important..."      │
│ 3. "Many families tell us..."       │
└─────────────────────────────────────┘
```

## 🗄️ Database Schema

**QuestDB Tables** (Time-series optimized):

| Table | Partition | Retention | Purpose |
|-------|-----------|-----------|---------|
| `csi_raw` | DAY | 7 days | Raw CSI (high-frequency) |
| `doppler_estimates` | DAY | 30 days | Velocity estimates |
| `trajectory_points` | DAY | 30 days | Subject movement |
| `movement_metrics` | HOUR | 30 days | Aggregated LMA features |
| `ocean_predictions` | HOUR | 90 days | Personality predictions |
| `handshake_matches` | DAY | 90 days | WiFi→Session links |
| `sales_intelligence` | HOUR | 1 year | Agent outputs |

**Example Queries**:

```sql
-- Latest OCEAN scores per subject
SELECT * FROM ocean_predictions
LATEST ON ts PARTITION BY subject_id;

-- Average hesitation by hour
SELECT ts, avg(hesitation_score)
FROM movement_metrics
WHERE subject_id = 'xyz'
SAMPLE BY 1h;

-- Join trajectory with metrics
SELECT t.ts, t.position_x, m.hesitation_score
FROM trajectory_points t
ASOF JOIN movement_metrics m ON(subject_id)
WHERE t.subject_id = 'xyz';
```

## 🔌 API Endpoints

### REST API (HTTP/2)

```
POST   /api/v1/sessions              Create new session
GET    /api/v1/sessions/:id          Get session data
POST   /api/v1/questionnaire          Submit questionnaire
GET    /api/v1/intelligence/:session  Get sales intelligence
GET    /api/v1/subjects/:id/trajectory Get trajectory
GET    /api/v1/subjects/:id/ocean     Get OCEAN prediction
GET    /api/v1/health                 Health check
```

### WebTransport (HTTP/3)

```
/wt/csi          Bidirectional stream: Raw CSI data
/wt/densepose    Bidirectional stream: Pose estimation
/wt/intelligence Bidirectional stream: Sales intelligence updates
```

## 🔒 Privacy & GDPR Compliance

### Privacy Principles

1. **Edge Processing**: Raw CSI processed locally, never stored
2. **Abstract Features**: Only kinematic/LMA features stored (not raw signals)
3. **Anonymous IDs**: Ephemeral UUIDs, unlinked to identity
4. **Data Minimization**: Auto-expire by table (7-90 day retention)
5. **Explicit Consent**: Kiosk interaction implies consent

### Data Flow

```
Raw CSI → [Edge Processing] → Abstract Features → Database
                                                  ↓
                                           (Auto-delete 7-90 days)
```

### GDPR Rights

- **Right to Access**: API endpoint to retrieve subject data
- **Right to Erasure**: Immediate deletion of subject records
- **Right to Object**: Opt-out via kiosk, stops tracking
- **Data Portability**: Export subject data in JSON format

## 📚 Research Citations

1. Koppensteiner & Grammer (2010) - Motion patterns in mating contexts
2. Satchell et al. (2017) - Movement velocity correlates with extraversion
3. Thoresen et al. (2012) - Nonverbal behavior and personality perception
4. Zhao et al. (2018) - RF-based pose estimation
5. Wang et al. (2021) - Cross-modal knowledge distillation for WiFi sensing

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📧 Contact

- **Author**: [cadieuxj](https://github.com/cadieuxj)
- **Organization**: Pairing Revolution
- **Email**: Contact via GitHub Issues

---

**⚠️ Research Prototype**: This system is a proof-of-concept for research purposes. Deployment in production requires:
- Proper WiFi hardware calibration
- Model training on real-world data
- Legal review for privacy compliance
- Ethical oversight for psychometric profiling
