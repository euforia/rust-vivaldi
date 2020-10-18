
use std::collections::HashMap;

use crate::config::Config;
use crate::utils::*;
use crate::coordinate::{Coordinate,DIMENSIONS_INCOMPATIBLE};

const MAX_RTT_SECS: f64 = 10.0;
const ZERO_SECS: f64 = 0.0;

const COORDINATE_INVALID: &str = "coordinate invalid";

// ClientStats is used to record events that occur when updating coordinates.
#[derive(Debug)]
pub struct ClientStats {
	// Resets is incremented any time we reset our local coordinate because
	// our calculations have resulted in an invalid state.
    resets: isize,
    // Successful updates
    updates: isize,
}

// Client manages the estimated network coordinate for a given node, and adjusts
// it as the node observes round trip times and estimated coordinates from other
// nodes. The core algorithm is based on Vivaldi, see the documentation for Config
// for more details.
#[derive(Debug)]
pub struct Client {
    // coord is the current estimate of the client's network coordinate.
    coord: Coordinate,
    // origin is a coordinate sitting at the origin.
    origin: Coordinate,
    // config contains the tuning parameters that govern the performance of
	// the algorithm.
    config: Config,
    // adjustmentIndex is the current index into the adjustmentSamples slice.
    adjustment_index: usize,
    // adjustmentSamples is used to store samples for the adjustment calculation.
    adjustment_samples: Vec<f64>,
    // latencyFilterSamples is used to store the last several RTT samples,
	// keyed by node name. We will use the config's LatencyFilterSamples
	// value to determine how many samples we keep, per node.
    latency_filter_samples: HashMap<String,Vec<f64>>,
    // stats is used to record events that occur when updating coordinates.
    _stats: ClientStats,
}

impl Client {
    // instantiates a new client with the Config. If the dimensionality is 
    // less than 1 is sets it to 1
    pub fn new(config: Config) -> Self {
        // Set the default dimensionality of 1
        let mut conf = config;
        if conf.dimensionality < 1 {
            conf.dimensionality = 1;
        }

        Client{
            adjustment_samples: vec![0.0; conf.adjustment_window_size],
            coord: Coordinate::new(&conf),
            origin: Coordinate::new(&conf),
            config: conf,
            adjustment_index: 0,
            latency_filter_samples: HashMap::new(),
            _stats: ClientStats{
                resets: 0,
                updates: 0,
            },
        }
    }

    // retuns a client using the default config
    pub fn default() -> Self {
        let conf = Config::default();
        Client{
            adjustment_samples: vec![0.0; conf.adjustment_window_size],
            coord: Coordinate::new(&conf),
            origin: Coordinate::new(&conf),
            config: conf,
            adjustment_index: 0,
            latency_filter_samples: HashMap::new(),
            _stats: ClientStats{
                resets: 0,
                updates: 0,
            },
        }
    }

    pub fn get_coordinate(&self) -> Coordinate {
        self.coord.clone()
    }

    pub fn set_coordinate(&mut self, coord: Coordinate) -> Result<(),String> {
        match self.check_coordinate(&coord) {
            Err(e) => { return Err(e); },
            _ => {
                // we do not clone as we own (diff from original implementation)
                self.coord = coord;
                Ok(())
            },
        }
    }

    pub fn forget_node(&mut self, node: &str) {
        self.latency_filter_samples.remove(node);
    }

    pub fn stats(&self) -> ClientStats {
        ClientStats{
            resets: self._stats.resets,
            updates: self._stats.updates,
        }
    }

    // distance_to returns the estimated RTT from the client's coordinate to 
    // other, the coordinate for another node.
    pub fn distance_to(&self, other: &Coordinate) -> Result<f64, String> {
        // c.mutex.RLock()
        // defer c.mutex.RUnlock()
        self.coord.distance_to(other)
    }

    // Update takes other, a coordinate for another node, and rtt, a round trip
    // time observation for a ping to that node, and updates the estimated position of
    // the client's coordinate. Returns the updated coordinate.
    // fn update(&self, node: String, other: &Coordinate, rtt: Duration) -> Coordinate {
    pub fn update(&mut self, node: String, other: &Coordinate, rtt_secs: f64) -> Result<Coordinate, String> {
        // c.mutex.Lock()
        // defer c.mutex.Unlock()

        match self.check_coordinate(other) {
            Err(e) => { return Err(e); },
            _ => {},
        };

        
        if rtt_secs <= ZERO_SECS || rtt_secs > MAX_RTT_SECS {
            return Err(
                format!("round trip time not in valid range, duration {:?} is not a positive value less than {:?} ", 
                    rtt_secs, MAX_RTT_SECS)
            )
        }

        let rtt_secs = self.latency_filter(node, rtt_secs);
        
        match self.update_vivaldi(other, rtt_secs) {
            Err(e) => { return Err(e); },
            _ => {},
        };

        self.update_adjustment(other, rtt_secs);

        match self.update_gravity() {
            Err(e) => { return Err(e); },
            _ => {},
        };
        
        if !self.coord.is_valid() {
            self._stats.resets += 1;
            self.coord = Coordinate::new(&self.config);
        }
        self._stats.updates += 1;

        Ok(self.coord.clone())
    }

    // checkCoordinate returns an error if the coordinate isn't compatible with
    // this client, or if the coordinate itself isn't valid. This assumes the 
    // mutex has been locked already.
    fn check_coordinate(&self, other: &Coordinate) -> Result<(),String> {
        if !self.coord.is_compatible_with(other) {
            return Err(DIMENSIONS_INCOMPATIBLE.to_string());
        }

        if !other.is_valid() {
            return Err(COORDINATE_INVALID.to_string());
        }

        Ok(())
    }

    // updateGravity applies a small amount of gravity to pull coordinates towards
    // the center of the coordinate system to combat drift. This assumes that the
    // mutex is locked already.
    fn update_gravity(&mut self) -> Result<(),String> {
        let dist = match self.origin.distance_to(&self.coord) {
            Ok(rtt) => { rtt },
            Err(e) => { 
                return Err(format!("update_gravity.distance_to: {:?}", e));
            },
        };

        let force = -1.0 * (dist/self.config.gravity_rho).powf(2.0);
        
        // self.coord = self.coord.apply_force(self.config.height_min, force, &self.origin)
        match self.coord.apply_force(self.config.height_min, force, &self.origin) {
            Ok(coord) => { 
                self.coord = coord;
                Ok(())
            },
            Err(e) => {
                return Err(format!("update_gravity.apply_force: {:?}", e));
            },
        }
    }


    // updateAdjustment updates the adjustment portion of the client's coordinate, if
    // the feature is enabled. This assumes that the mutex has been locked already.
    fn update_adjustment(&mut self, other: &Coordinate, rtt_secs: f64) {
        if self.config.adjustment_window_size == 0 {
            return
        }
    
        // Note that the existing adjustment factors don't figure in to this
        // calculation so we use the raw distance here.
        let dist = self.coord.raw_distance_to(other);
        self.adjustment_samples[self.adjustment_index] = rtt_secs - dist;
        self.adjustment_index = (self.adjustment_index + 1) % self.config.adjustment_window_size;
    
        let mut sum = 0.0;
        for sample in self.adjustment_samples.iter() {
            sum += sample
        }

        self.coord.adjustment = sum / (2.0 * (self.config.adjustment_window_size as f64))
    }

    // latencyFilter applies a simple moving median filter with a new sample for
    // a node. This assumes that the mutex has been locked already.
    fn latency_filter(&mut self, node: String, rtt_secs: f64) -> f64 {
        // 	samples, ok := c.latencyFilterSamples[node]
        // 	if !ok {
        // 		samples = make([]float64, 0, c.config.LatencyFilterSize)
        // 	}

        // No sample so we add the sample and return the new sample
        if !self.latency_filter_samples.contains_key(&node) {
            self.latency_filter_samples.insert(node.to_string(), vec![rtt_secs]);
            return rtt_secs;
        }

        let samples = self.latency_filter_samples.get_mut(&node).unwrap();
    	// Add the new sample and trim the list, if needed.
        samples.push(rtt_secs);
    
        if samples.len() > self.config.latency_filter_size {
            // samples = samples[1:]
            samples.remove(0);
    	}
        
        // assign ... 
        // 	c.latencyFilterSamples[node] = samples
        // self.latency_filter_samples.insert(node, samples.to_vec());

    	// Sort a copy of the samples and return the median.
        let mut sorted = vec![0.0; samples.len()];
        for i in 0..samples.len() {
            sorted[i] = samples[i];
        }
        sorted.sort_by(cmp_f64);

        sorted[sorted.len()/2]
    }

    fn update_vivaldi(&mut self, other: &Coordinate, rtt_seconds: f64) -> Result<(),String> {
        let mut rtt_secs = rtt_seconds;
        if rtt_secs < ZERO_THRESHOLD {
            rtt_secs = ZERO_THRESHOLD
        }

        let dist = match self.coord.distance_to(other){
            Ok(rtt) => { rtt },
            Err(e) => {
                return Err(format!("update_vivaldi.distance_to: {:?}", e));
                // return Err(e);
            },
        };
        
        let wrongness = (dist-rtt_secs).abs() / rtt_secs;
    
        let mut total_error = self.coord.error + other.error;
        if total_error < ZERO_THRESHOLD {
            total_error = ZERO_THRESHOLD;
        }
        let weight = self.coord.error / total_error;
    
        self.coord.error = self.config.vivaldi_ce * weight * wrongness + 
            self.coord.error*(1.0-self.config.vivaldi_ce*weight);
        
        if self.coord.error > self.config.vivaldi_error_max {
            self.coord.error = self.config.vivaldi_error_max
        }
    
        let delta = self.config.vivaldi_cc * weight;
        let force = delta * (rtt_secs - dist);
        
        // self.coord = self.coord.apply_force(self.config.height_min, force, other);
        match self.coord.apply_force(self.config.height_min, force, other) {
            Ok(coord) => { 
                self.coord = coord;
                Ok(())
            },
            Err(e) => { 
                return Err(format!("update_vivaldi.apply_force: {:?}", e));
                // return Err(e);
            },
        }
    }

}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_client_new_0_dimensionality() {
        let mut c1 = Config::default();
        c1.dimensionality = 0;
        let client = Client::new(c1);//.unwrap_err();
        assert_eq!(1, client.config.dimensionality);
    }

    #[test]
    fn test_client_new() {
        let c2 = Config::default();
        let origin = Coordinate::new(&c2);
        let client = Client::new(c2);//.unwrap();
        assert_eq!(client.get_coordinate(), origin,
            "fresh client should be located at the origin");
    }

    #[test]
    fn test_client_default() {
        let client = Client::default();
        let origin = Coordinate::new(&client.config);
        assert_eq!(client.get_coordinate(), origin,
            "fresh client should be located at the origin");
    }

    #[test]
    fn test_client_distance_to() {
        let mut config = Config::default();
        config.dimensionality = 3;
        config.height_min = 0.0;
    
        // Fiddle a raw coordinate to put it a specific number of seconds away.
        let mut other = Coordinate::new(&config);
        other.vec[2] = 12.345;

        let client = Client::new(config);//.unwrap();

        assert_eq!(client.distance_to(&other).unwrap(), other.vec[2]);
    }

    #[test]
    fn test_client_latency_filter() {
        let mut config = Config::default();
        config.latency_filter_size = 3;
    
        let mut client = Client::new(config);//.unwrap();
    
        // Make sure we get the median, and that things age properly.
        assert_eq!(client.latency_filter("alice".to_string(), 0.201), 0.201);
        assert_eq!(client.latency_filter("alice".to_string(), 0.200), 0.201);
        assert_eq!(client.latency_filter("alice".to_string(), 0.207), 0.201);
    
        // This glitch will get median-ed out and never seen by Vivaldi.
        assert_eq!(client.latency_filter("alice".to_string(), 1.9), 0.207);
        assert_eq!(client.latency_filter("alice".to_string(), 0.203), 0.207);
        assert_eq!(client.latency_filter("alice".to_string(), 0.199), 0.203);
        assert_eq!(client.latency_filter("alice".to_string(), 0.211), 0.203);
    
        // Make sure different nodes are not coupled.
        assert_eq!(client.latency_filter("bob".to_string(), 0.310), 0.310);
    
        // Make sure we don't leak coordinates for nodes that leave.
        client.forget_node("alice");
        assert_eq!(client.latency_filter("alice".to_string(), 0.888), 0.888);
    }

    #[test]
    fn test_client_nan_defense() {
        let mut config = Config::default();
        config.latency_filter_size = 3;

        let mut other = Coordinate::new(&config);
    
        let mut client = Client::new(config);//.unwrap();
    
        // Block a bad coordinate from coming in.
        other.vec[0] = f64::NAN;
        assert!(!other.is_valid());

        let rtt = 0.250;
        // let rtt = Duration::from_millis(250);
        
        let mut err = client.update("node".to_string(), &other, rtt).unwrap_err();
        assert!(err.contains(COORDINATE_INVALID));
        assert!(client.get_coordinate().is_valid());
    
        // Block setting an invalid coordinate directly.
        err = client.set_coordinate(other.clone()).unwrap_err();
        assert!(err.contains(COORDINATE_INVALID));
        assert!(client.get_coordinate().is_valid());
    
        // Block an incompatible coordinate.
        other.vec = vec![0.0; 2*other.vec.len()]; 
        err = client.update("node".to_string(), &other, rtt).unwrap_err();
        assert!(err.contains(DIMENSIONS_INCOMPATIBLE));
        assert!(client.get_coordinate().is_valid());

    
        // Block setting an incompatible coordinate directly.
        err = client.set_coordinate(other.clone()).unwrap_err();
        assert!(err.contains("dimensions incompatible"));
        assert!(client.get_coordinate().is_valid());

        // Poison the internal state and make sure we reset on an update.
        client.coord.vec[0] = f64::NAN;
        // other = Coordinate::new(&config);
        let pc = client.update("node".to_string(), &Coordinate::new(&Config::default()), rtt).unwrap();
        assert!(pc.is_valid());
        assert_eq!(1, client.stats().resets);
    }

    #[test]
    fn test_client_update() {
        let mut config = Config::default();
        config.dimensionality = 3;

        let mut other = Coordinate::new(&config);

        let mut client = Client::new(config);//.unwrap();
    
        // Make sure the Euclidean part of our coordinate is what we expect.
        let c = client.get_coordinate();
        assert_eq!(c.vec, vec![0.0, 0.0, 0.0]);
    
        // Place a node right above the client and observe an RTT longer than the
        // client expects, given its distance.
        other.vec[2] = 0.001;
        let rtt = 2.0 * other.vec[2];
        // let rtt = Duration::from_secs_f64(2.0 * other.vec[2]);
        let mut uc = client.update("node".to_string(), &other, rtt).unwrap();
   
        // The client should have scooted down to get away from it.
        assert!(uc.vec[2] < 0.0, format!(
            "client z coordinate {:.6?} should be < 0.0", uc.vec[2]
        ));
    
        // Set the coordinate to a known state.
        uc.vec[2] = 99.0;
        client.set_coordinate(uc).unwrap();
        let gc = client.get_coordinate();
        assert_eq!(gc.vec[2], 99.0);

        // Check update counter
        assert_eq!(1, client.stats().updates);
    }

    #[test]
    fn test_client_invalid_in_ping_values() {
        let mut config = Config::default();
        config.dimensionality = 3;
    
        let mut other = Coordinate::new(&config);

        let mut client = Client::new(config);//.unwrap();
    
        // Place another node
        other.vec[2] = 0.001;
        let dist = client.distance_to(&other).unwrap();
    
        // Update with a series of invalid ping periods, should return an error 
        // and estimated rtt remains unchanged
        // pings := []int{1<<63 - 1, -35, 11}
        let pings = vec![-1.0, -35.0, -11.0];

        for ping in pings.iter() {
            client.update("node".to_string(), &other, *ping).unwrap_err();
    
            let dist_new = client.distance_to(&other).unwrap();
            assert_eq!(dist_new, dist); 
        }
    }
}