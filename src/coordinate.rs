use crate::config::Config;
use crate::utils::*;

pub const DIMENSIONS_INCOMPATIBLE: &str = "dimensions incompatible";

// Coordinate is a specialized structure for holding network coordinates for the
// Vivaldi-based coordinate mapping algorithm. All of the fields should be public
// to enable this to be serialized. All values in here are in units of seconds.
#[derive(Debug, PartialEq, PartialOrd)]
pub struct Coordinate {
    // Vec is the Euclidean portion of the coordinate. This is used along
    // with the other fields to provide an overall distance estimate. The
    // units here are seconds.
    pub vec: Vec<f64>,
    // Err reflects the confidence in the given coordinate and is updated
    // dynamically by the Vivaldi Client. This is dimensionless.
    pub error: f64,
    // Adjustment is a distance offset computed based on a calculation over
    // observations from all other nodes over a fixed window and is updated
    // dynamically by the Vivaldi Client. The units here are seconds.
    pub adjustment: f64,
    // Height is a distance offset that accounts for non-Euclidean effects
    // which model the access links from nodes to the core Internet. The access
    // links are usually set by bandwidth and congestion, and the core links
    // usually follow distance based on geography.
    pub height: f64,
}

impl Clone for Coordinate {
    fn clone(&self) -> Coordinate {
        let mut v = vec![0.0; self.vec.len()];
        for i in 0..self.vec.len() {
            v[i] = self.vec[i]
        }

        Coordinate {
            vec: v,
            error: self.error,
            adjustment: self.adjustment,
            height: self.height,
        }
    }
}

impl Coordinate {
    pub fn new(config: &Config) -> Self {
        Coordinate {
            vec: vec![0.0; config.dimensionality],
            error: config.vivaldi_error_max,
            adjustment: 0.0,
            height: config.height_min,
        }
    }

    // is_compatible_with checks to see if the two coordinates are compatible
    // dimensionally. If this returns true then you are guaranteed to not get
    // any runtime errors operating on them.
    pub fn is_compatible_with(&self, other: &Coordinate) -> bool {
        self.vec.len() == other.vec.len()
    }

    pub fn apply_force(
        &self,
        height_min: f64,
        force: f64,
        other: &Coordinate,
    ) -> Result<Coordinate, String> {
        if !self.is_compatible_with(other) {
            return Err(DIMENSIONS_INCOMPATIBLE.to_string());
        }

        let mut ret = self.clone();

        let (unit, mag) = unit_vector_at(&self.vec, &other.vec);

        ret.vec = add(&ret.vec, &mul(&unit, force));

        if mag > ZERO_THRESHOLD {
            ret.height = (ret.height + other.height) * force / mag + ret.height;
            ret.height = ret.height.max(height_min);
        }

        Ok(ret)
    }

    pub fn is_valid(&self) -> bool {
        for i in 0..self.vec.len() {
            if !component_is_valid(self.vec[i]) {
                return false;
            }
        }

        component_is_valid(self.error)
            && component_is_valid(self.adjustment)
            && component_is_valid(self.height)
    }

    // It returns the distance is seconds
    pub fn distance_to(&self, other: &Coordinate) -> Result<f64, String> {
        if !self.is_compatible_with(other) {
            return Err(DIMENSIONS_INCOMPATIBLE.to_string());
        }

        let mut dist = self.raw_distance_to(other);
        let adjust_dist = dist + self.adjustment + other.adjustment;
        if adjust_dist > 0.0 {
            dist = adjust_dist;
        }
        Ok(dist)
    }

    pub fn raw_distance_to(&self, other: &Coordinate) -> f64 {
        magnitude(&diff(&self.vec, &other.vec)) + self.height + other.height
    }
}

// component_is_valid returns false if a floating point value is a NaN or an
// infinity.
fn component_is_valid(f: f64) -> bool {
    !f.is_infinite() && !f.is_nan()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new() {
        let config = Config::default();
        let c = Coordinate::new(&config);
        assert_eq!(
            c.vec.len(),
            config.dimensionality,
            "dimensionality incorrect"
        );
    }

    #[test]
    fn test_clone() {
        let mut c = Coordinate::new(&Config::default());
        c.vec[0] = 1.0;
        c.vec[1] = 2.0;
        c.vec[2] = 3.0;

        c.error = 5.0;
        c.adjustment = 10.0;
        c.height = 4.2;

        let mut other = c.clone();
        assert_eq!(c, other, "clone invalid");

        other.vec[0] = c.vec[0] + 0.5;
        assert_ne!(c, other, "clone pointing to ancestor");
    }

    #[test]
    fn test_is_compatible_with() {
        let mut config = Config::default();
        config.dimensionality = 3;

        let c1 = Coordinate::new(&config);
        let c2 = Coordinate::new(&config);

        config.dimensionality = 2;
        let alien = Coordinate::new(&config);

        assert!(
            c1.is_compatible_with(&c1)
                && c2.is_compatible_with(&c2)
                && alien.is_compatible_with(&alien),
            "coordinates should be compatible with themselves"
        );

        // if !c1.is_compatible_with(&c1) || !c2.is_compatible_with(&c2) ||
        //     !alien.is_compatible_with(&alien) {
        //     t.Fatalf("coordinates should be compatible with themselves")
        // }

        assert!(
            c1.is_compatible_with(&c2) && c2.is_compatible_with(&c1),
            "coordinates should be compatible with each other"
        );
        // if !c1.is_compatible_with(&c2) || !c2.is_compatible_with(&c1) {
        //     t.Fatalf("coordinates should be compatible with each other")
        // }

        assert!(
            !c1.is_compatible_with(&alien)
                && !c2.is_compatible_with(&alien)
                && !alien.is_compatible_with(&c1)
                && !alien.is_compatible_with(&c2),
            "alien should not be compatible with the other coordinates"
        );

        // if c1.is_compatible_with(&alien) || c2.is_compatible_with(&alien) ||
        //     alien.is_compatible_with(&c1) || alien.is_compatible_with(&c2) {
        //     t.Fatalf("alien should not be compatible with the other coordinates")
        // }
    }

    #[test]
    fn test_is_valid() {
        let mut c = Coordinate::new(&Config::default());

        // vec
        for i in 0..c.vec.len() {
            assert!(c.is_valid());

            c.vec[i] = f64::NAN;
            assert!(!c.is_valid(), "vector should not be valid {:?}", i);

            c.vec[i] = 0.0;
            assert!(c.is_valid(), "field should be valid {:?}", i);

            c.vec[i] = f64::INFINITY;
            assert!(!c.is_valid(), "vector should not be valid {:?}", i);

            c.vec[i] = 0.0;
            assert!(c.is_valid(), "field should be valid {:?}", i);
        }

        // error
        assert!(c.is_valid());
        c.error = f64::NAN;
        assert!(!c.is_valid(), "error should not be valid");
        c.error = 0.0;
        assert!(c.is_valid(), "error should be valid");
        c.error = f64::INFINITY;
        assert!(!c.is_valid(), "error should not be valid");
        c.error = 0.0;
        assert!(c.is_valid(), "error should be valid");

        // adjustment
        assert!(c.is_valid());
        c.adjustment = f64::NAN;
        assert!(!c.is_valid(), "adjustment should not be valid");
        c.adjustment = 0.0;
        assert!(c.is_valid(), "adjustment should be valid");
        c.adjustment = f64::INFINITY;
        assert!(!c.is_valid(), "adjustment should not be valid");
        c.adjustment = 0.0;
        assert!(c.is_valid(), "adjustment should be valid");

        // height
        assert!(c.is_valid());
        c.height = f64::NAN;
        assert!(!c.is_valid(), "height should not be valid");
        c.height = 0.0;
        assert!(c.is_valid(), "height should be valid");
        c.height = f64::INFINITY;
        assert!(!c.is_valid(), "height should not be valid");
        c.height = 0.0;
        assert!(c.is_valid(), "height should be valid");
    }

    #[test]
    fn test_apply_force() {
        let mut config = Config::default();
        config.dimensionality = 3;
        config.height_min = 0.0;

        let mut origin = Coordinate::new(&config);

        // This proves that we normalize, get the direction right, and apply the
        // force multiplier correctly.
        let mut above = Coordinate::new(&config);
        above.vec = vec![0.0, 0.0, 2.9];
        let mut c = origin.apply_force(config.height_min, 5.3, &above).unwrap();
        assert_eq!(c.vec, vec![0.0, 0.0, -5.3]);

        // Scoot a point not starting at the origin to make sure there's nothing
        // special there.
        let mut right = Coordinate::new(&config);
        right.vec = vec![3.4, 0.0, -5.3];
        c = c.apply_force(config.height_min, 2.0, &right).unwrap();
        assert_eq!(c.vec, vec![-2.0, 0.0, -5.3]);

        // If the points are right on top of each other, then we should end up
        // in a random direction, one unit away. This makes sure the unit vector
        // build up doesn't divide by zero.
        c = origin.apply_force(config.height_min, 1.0, &origin).unwrap();
        // Here we check after rounding the distance as it may be between
        // 0.99999999999 and 1.000000000X.
        // assert_eq!(origin.distance_to(&c).unwrap().round(), 1.0);
        let pd = origin.distance_to(&c).unwrap();
        assert!(pd >= 0.9999999999999999 && pd <= 1.0000000000000002);
        // assert_eq!(origin.distance_to(&c).unwrap(), 1.0);

        // Enable a minimum height and make sure that gets factored in properly.
        config.height_min = 10.0e-6;
        origin = Coordinate::new(&config);
        c = origin.apply_force(config.height_min, 5.3, &above).unwrap();
        assert_eq!(c.vec, vec![0.0, 0.0, -5.3]);
        assert_eq!(c.height, config.height_min + 5.3 * config.height_min / 2.9);

        // Make sure the height minimum is enforced.
        c = origin.apply_force(config.height_min, -5.3, &above).unwrap();
        assert_eq!(c.vec, vec![0.0, 0.0, 5.3]);
        assert_eq!(c.height, config.height_min);

        // Shenanigans should get called if the dimensions don't match.
        let mut bad = c.clone();
        bad.vec = vec![0.0; bad.vec.len() + 1];
        c.apply_force(config.height_min, 1.0, &bad).unwrap_err();
        // verifyDimensionPanic(t, func() { c.ApplyForce(config, 1.0, bad) })
    }

    #[test]
    fn test_distance_to() {
        let mut config = Config::default();
        config.dimensionality = 3;
        config.height_min = 0.0;

        let mut c1 = Coordinate::new(&config);
        let mut c2 = Coordinate::new(&config);

        c1.vec = vec![-0.5, 1.3, 2.4];
        c2.vec = vec![1.2, -2.3, 3.4];

        assert_eq!(c1.distance_to(&c1).unwrap(), 0.0);
        assert_eq!(c1.distance_to(&c2).unwrap(), c2.distance_to(&c1).unwrap());
        assert_eq!(c1.distance_to(&c2).unwrap(), 4.104875150354758);

        // Make sure negative adjustment factors are ignored.
        c1.adjustment = -1.0e6;
        assert_eq!(c1.distance_to(&c2).unwrap(), 4.104875150354758);

        // Make sure positive adjustment factors affect the distance.
        c1.adjustment = 0.1;
        c2.adjustment = 0.2;
        assert_eq!(c1.distance_to(&c2).unwrap(), 4.104875150354758 + 0.3);

        // Make sure the heights affect the distance.
        c1.height = 0.7;
        c2.height = 0.1;
        assert_eq!(c1.distance_to(&c2).unwrap(), 4.104875150354758 + 0.3 + 0.8);

        // Shenanigans should get called if the dimensions don't match.
        let mut bad = c1.clone();
        bad.vec = vec![0.0; bad.vec.len() + 1];
        c1.distance_to(&bad).unwrap_err();
        // verifyDimensionPanic(t, func() { _ = c1.DistanceTo(bad) })
    }

    #[test]
    fn test_raw_distance_to() {
        let mut config = Config::default();
        config.dimensionality = 3;
        config.height_min = 0.0;

        let mut c1 = Coordinate::new(&config);
        let mut c2 = Coordinate::new(&config);

        c1.vec = vec![-0.5, 1.3, 2.4];
        c2.vec = vec![1.2, -2.3, 3.4];

        assert_eq!(c1.raw_distance_to(&c1), 0.0);
        assert_eq!(c1.raw_distance_to(&c2), c2.raw_distance_to(&c1));
        assert_eq!(c1.raw_distance_to(&c2), 4.104875150354758);

        // Make sure that the adjustment doesn't factor into the raw
        // distance.
        c1.adjustment = 1.0e6;
        assert_eq!(c1.raw_distance_to(&c2), 4.104875150354758);

        // Make sure the heights affect the distance.
        c1.height = 0.7;
        c2.height = 0.1;
        assert_eq!(c1.raw_distance_to(&c2), 4.104875150354758 + 0.8);
    }

    // distance is a self-contained example that appears in documentation. It
    // returns the distance is seconds
    fn distance(a: &Coordinate, b: &Coordinate) -> f64 {
        // Coordinates will always have the same dimensionality, so this is
        // just a sanity check.
        if a.vec.len() != b.vec.len() {
            panic!("dimensions aren't compatible")
        }

        // Calculate the Euclidean distance plus the heights.
        let mut sumsq = 0.0;
        for i in 0..a.vec.len() {
            let diff = a.vec[i] - b.vec[i];
            sumsq += diff * diff
        }

        let mut rtt = sumsq.sqrt() + a.height + b.height;

        // Apply the adjustment components, guarding against negatives.
        let adjusted = rtt + a.adjustment + b.adjustment;
        if adjusted > 0.0 {
            rtt = adjusted;
        }

        rtt
    }

    #[test]
    fn test_dist_example() {
        let config = Config::default();

        let mut c1 = Coordinate::new(&config);
        let mut c2 = Coordinate::new(&config);

        c1.vec = vec![-0.5, 1.3, 2.4];
        c2.vec = vec![1.2, -2.3, 3.4];

        c1.adjustment = 0.1;
        c2.adjustment = 0.2;

        c1.height = 0.7;
        c2.height = 0.1;

        assert_eq!(c1.distance_to(&c2).unwrap(), distance(&c1, &c2));
    }

    #[test]
    fn test_add() {
        let vec1 = vec![1.0, -3.0, 3.0];
        let vec2 = vec![-4.0, 5.0, 6.0];
        assert_eq!(add(&vec1, &vec2), vec![-3.0, 2.0, 9.0]);

        let zero = vec![0.0, 0.0, 0.0];
        assert_eq!(add(&vec1, &zero), vec1);
    }

    #[test]
    fn test_diff() {
        let vec1 = vec![1.0, -3.0, 3.0];
        let vec2 = vec![-4.0, 5.0, 6.0];
        assert_eq!(diff(&vec1, &vec2), vec![5.0, -8.0, -3.0]);

        let zero = vec![0.0, 0.0, 0.0];
        assert_eq!(diff(&vec1, &zero), vec1);
    }

    #[test]
    fn test_magnitude() {
        let zero = vec![0.0, 0.0, 0.0];
        assert_eq!(magnitude(&zero), 0.0);

        let vec = vec![1.0, -2.0, 3.0];
        assert_eq!(magnitude(&vec), 3.7416573867739413);
    }

    #[test]
    fn test_unit_vector_at() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![0.5, 0.6, 0.7];
        let (u, mag) = unit_vector_at(&vec1, &vec2);

        assert_eq!(
            u,
            vec![0.18257418583505536, 0.511207720338155, 0.8398412548412546]
        );
        // Due to floating point precision, the value could be 1.0
        assert!(
            magnitude(&u) == 0.9999999999999999 || magnitude(&u) == 1.0,
            "got: {:.15}",
            magnitude(&u)
        );

        assert_eq!(mag, magnitude(&diff(&vec1, &vec2)));

        // If we give positions that are equal we should get a random unit vector
        // returned to us, rather than a divide by zero.
        let (u, mag) = unit_vector_at(&vec1, &vec1);
        assert!(magnitude(&u) == 0.9999999999999999 || magnitude(&u) == 1.0);
        assert_eq!(mag, 0.0);

        // We can't hit the final clause without heroics so I manually forced it
        // there to verify it works.
    }
}
