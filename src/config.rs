
#[derive(Debug)]
pub struct Config  {
    // The dimensionality of the coordinate system. As discussed in [2], more
	// dimensions improves the accuracy of the estimates up to a point. Per [2]
	// we chose 8 dimensions plus a non-Euclidean height.
	pub dimensionality: usize,

	// VivaldiErrorMax is the default error value when a node hasn't yet made
	// any observations. It also serves as an upper limit on the error value in
	// case observations cause the error value to increase without bound.
	pub vivaldi_error_max: f64,

	// VivaldiCE is a tuning factor that controls the maximum impact an
	// observation can have on a node's confidence. See [1] for more details.
	pub vivaldi_ce: f64,

	// VivaldiCC is a tuning factor that controls the maximum impact an
	// observation can have on a node's coordinate. See [1] for more details.
	pub vivaldi_cc: f64,

	// AdjustmentWindowSize is a tuning factor that determines how many samples
	// we retain to calculate the adjustment factor as discussed in [3]. Setting
	// this to zero disables this feature.
	pub adjustment_window_size: usize,

	// HeightMin is the minimum value of the height parameter. Since this
	// always must be positive, it will introduce a small amount error, so
	// the chosen value should be relatively small compared to "normal"
	// coordinates.
	pub height_min: f64,

	// LatencyFilterSamples is the maximum number of samples that are retained
	// per node, in order to compute a median. The intent is to ride out blips
	// but still keep the delay low, since our time to probe any given node is
	// pretty infrequent. See [2] for more details.
	pub latency_filter_size: usize,

	// GravityRho is a tuning factor that sets how much gravity has an effect
	// to try to re-center coordinates. See [2] for more details.
	pub gravity_rho: f64,
}

impl Config {
    pub fn default() -> Self {
        Config{
            dimensionality:         8,
            vivaldi_error_max:      1.5,
            vivaldi_ce:             0.25,
            vivaldi_cc:             0.25,
            adjustment_window_size: 20,
            height_min:             10.0e-6,
            latency_filter_size:    3,
            gravity_rho:            150.0,
        }
	}
}