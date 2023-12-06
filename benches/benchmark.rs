use criterion::{criterion_group, criterion_main, Criterion};
use llamars::generate;

pub fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark-group");
    group.sample_size(10).bench_function("benchmark", |b| {
        b.iter(|| {
            generate("weights.bin".into(), "Once upon a time".into(), 256, true).unwrap();
        })
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
