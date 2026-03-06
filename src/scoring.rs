#[derive(Clone, Copy)]
pub struct ScoreWeights {
    pub coverage: f64,
    pub structure: f64,
    pub info_density: f64,
    pub noise: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            coverage: 0.30,
            structure: 0.25,
            info_density: 0.25,
            noise: 0.20,
        }
    }
}

pub fn clamp_0_100(value: f64) -> f64 {
    value.clamp(0.0, 100.0)
}

pub fn dqi_total(
    coverage_score: f64,
    structure_score: f64,
    info_density_score: f64,
    noise_score: f64,
    weights: ScoreWeights,
) -> f64 {
    clamp_0_100(
        coverage_score * weights.coverage
            + structure_score * weights.structure
            + info_density_score * weights.info_density
            + noise_score * weights.noise,
    )
}
