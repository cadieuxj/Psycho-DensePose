//! Benchmarks for CSI processing pipeline.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex;

use psycho_csi::packet::CsiPacket;
use psycho_csi::sanitizer::CsiSanitizer;

fn create_test_packet(n_subcarriers: u16) -> CsiPacket {
    let csi_matrix: Vec<Complex<f64>> = (0..n_subcarriers)
        .map(|i| {
            let phase = (i as f64 * 0.01) + 0.05;
            Complex::from_polar(1.0, phase)
        })
        .collect();

    CsiPacket {
        timestamp: 0,
        antenna_id: 0,
        subcarrier_count: n_subcarriers,
        csi_matrix,
        rssi: -45,
        noise_floor: -90,
        tx_mac_hash: 0,
        sequence_number: 0,
    }
}

fn benchmark_sanitization(c: &mut Criterion) {
    let sanitizer = CsiSanitizer::new();

    let packet_160mhz = create_test_packet(1992);
    let packet_80mhz = create_test_packet(996);
    let packet_20mhz = create_test_packet(242);

    c.bench_function("sanitize_160mhz", |b| {
        b.iter(|| sanitizer.sanitize(black_box(&packet_160mhz)))
    });

    c.bench_function("sanitize_80mhz", |b| {
        b.iter(|| sanitizer.sanitize(black_box(&packet_80mhz)))
    });

    c.bench_function("sanitize_20mhz", |b| {
        b.iter(|| sanitizer.sanitize(black_box(&packet_20mhz)))
    });
}

fn benchmark_hampel(c: &mut Criterion) {
    let sanitizer = CsiSanitizer::new();

    let data: Vec<f64> = (0..1992)
        .map(|i| (i as f64 * 0.01).sin() + if i == 500 { 100.0 } else { 0.0 })
        .collect();

    c.bench_function("hampel_filter_1992", |b| {
        b.iter(|| sanitizer.hampel_filter(black_box(&data)))
    });
}

fn benchmark_phase_unwrap(c: &mut Criterion) {
    let sanitizer = CsiSanitizer::new();

    let mut phase: Vec<f64> = (0..1992)
        .map(|i| ((i as f64 * 0.05) % (2.0 * std::f64::consts::PI)) - std::f64::consts::PI)
        .collect();

    c.bench_function("phase_unwrap_1992", |b| {
        b.iter(|| {
            let mut p = phase.clone();
            sanitizer.unwrap_phase(black_box(&mut p));
            p
        })
    });
}

criterion_group!(
    benches,
    benchmark_sanitization,
    benchmark_hampel,
    benchmark_phase_unwrap
);
criterion_main!(benches);
