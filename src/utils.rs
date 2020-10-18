
use rand::Rng;
use std::cmp::Ordering;

// ZERO_THRESHOLD is used to decide if two coordinates are on top of each
// other.
pub const ZERO_THRESHOLD: f64 = 1.0e-6;

// this assumes a and b are not infinite and not nan
pub fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

// unitVectorAt returns a unit vector pointing at vec1 from vec2. If the two
// positions are the same then a random unit vector is returned. We also return
// the distance between the points for use in the later height calculation.
pub fn unit_vector_at(vec1: &[f64], vec2: &[f64]) -> (Vec<f64>, f64) {
	let mut ret = diff(vec1, vec2);

    // If the coordinates aren't on top of each other we can normalize.
    let mut mag = magnitude(&ret);
    if mag > ZERO_THRESHOLD {
		return (mul(&ret, 1.0/mag), mag)
	}

	// Otherwise, just return a random unit vector.
    let mut rng = rand::thread_rng();
    for i in 0..ret.len() {
        ret[i] = rng.gen();   
    }
    mag = magnitude(&ret);
	if mag > ZERO_THRESHOLD {
		return (mul(&ret, 1.0/mag), 0.0)
	}

	// And finally just give up and make a unit vector along the first
	// dimension. This should be exceedingly rare.
    ret = vec![0.0; ret.len()];
	ret[0] = 1.0;
    (ret, 0.0)
}


// add returns the sum of vec1 and vec2. This assumes the dimensions have
// already been checked to be compatible.
pub fn add(vec1: &[f64], vec2: &[f64]) -> Vec<f64> {
    let mut ret = vec![0.0; vec1.len()];
    for i in 0..ret.len() {
		ret[i] = vec1[i] + vec2[i]
	}
	ret
}

// diff returns the difference between the vec1 and vec2. This assumes the
// dimensions have already been checked to be compatible.
pub fn diff(vec1: &[f64], vec2: &[f64]) -> Vec<f64> {
    let mut ret = vec![0.0; vec1.len()];
    for i in 0..ret.len() {
		ret[i] = vec1[i] - vec2[i]
	}
	ret
}

// mul returns vec multiplied by a scalar factor.
pub fn mul(vec1: &[f64], factor: f64) -> Vec<f64> {
    let mut ret = vec![0.0; vec1.len()];
	for i in 0..ret.len() {
		ret[i] = vec1[i] * factor
	}
    ret
}

// magnitude computes the magnitude of the vec.
pub fn magnitude(v: &[f64]) -> f64 {
	let mut sum = 0.0;
	for val in v.iter() {
		sum += val * val;
	}
    sum.sqrt()
}
