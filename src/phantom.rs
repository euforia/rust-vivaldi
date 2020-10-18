
#[cfg(test)]
mod test {
	use std::f64;

	use rand::Rng;
	use rand_distr::{Normal,Distribution};

	use crate::config::Config;
	use crate::client::Client;
	use crate::utils::magnitude;

	// Stats is returned from the Evaluate function with a summary of the algorithm
	// performance.
	#[derive(Debug)]
	struct Stats {
		error_max: f64,
		error_avg: f64,
	}

	// generate_Clients returns a slice with nodes number of clients, all with the
	// given config.
	fn generate_clients(n: usize, config: Config) -> Vec<Client> {
		let mut clients: Vec<Client> = Vec::new();

		for _ in 0..n {
			clients.push(Client::new(Config{
				dimensionality: config.dimensionality,
				vivaldi_error_max: config.vivaldi_error_max,
				vivaldi_ce: config.vivaldi_ce,
				vivaldi_cc: config.vivaldi_cc,
				adjustment_window_size: config.adjustment_window_size,
				height_min: config.height_min,
				latency_filter_size: config.latency_filter_size,
				gravity_rho: config.gravity_rho,
			}))
		}

		clients
	}

	// Simulate runs the given number of cycles using the given list of clients and
	// truth matrix. On each cycle, each client will pick a random node and observe
	// the truth RTT, updating its coordinate estimate. The RNG is re-seeded for
	// each simulation run to get deterministic results (for this algorithm and the
	// underlying algorithm which will use random numbers for position vectors when
	// starting out with everything at the origin).
	fn simulate(clients: &mut Vec<Client>, truth: &Vec<Vec<f64>>, cycles: isize) {
		// rand.Seed(1)
		let mut rng = rand::thread_rng();

		let n = clients.len();
		for _ in 0..cycles {
			for i in 0..n {
				let j = rng.gen_range(0, n);

				if j != i {
					let c = clients[j].get_coordinate();
					let rtt = truth[i][j];
					clients[i].update(format!("node_{:}", j), &c, rtt).unwrap();
				}
			}
		}
	}

	// Evaluate uses the coordinates of the given clients to calculate estimated
	// distances and compares them with the given truth matrix, returning summary
	// stats.
	fn evaluate(clients: Vec<Client>, truth: &Vec<Vec<f64>>) -> Stats {
		let mut stats = Stats{
			error_max: 0.0,
			error_avg: 0.0,
		};
		let n = clients.len();
		let mut count: f64 = 0.0;
		
		for i in 0..n {
			for j in 0..n {
				let est = clients[i].distance_to(&clients[j].get_coordinate()).unwrap();
				let actual = truth[i][j];
				let error = (est-actual).abs() / actual;
				// stats.error_max = math.Max(stats.error_max, error)
				stats.error_max = stats.error_max.max(error);
				stats.error_avg += error;
				count += 1.0;
			}
		}

		stats.error_avg /= count;
		println!("Error avg={:.15} max={:.15}\n", stats.error_avg, stats.error_max);
		stats
	}

	// generate_Line returns a truth matrix as if all the nodes are in a straight linke
	// with the given spacing between them.
	fn generate_line(n: usize, spacing: f64) -> Vec<Vec<f64>> {
		let mut truth: Vec<Vec<f64>> = Vec::new();
		for _ in 0..n {
			truth.push(vec![0.0; n]);
		}

		for i in 0..n {
			for j in i+1..n {
				let rtt = (j-i) as f64 * spacing;
				truth[i][j] = rtt;
				truth[j][i] = rtt;
			}
		}
		
		truth
	}

	// generate_Circle returns a truth matrix for a set of nodes, evenly distributed
	// around a circle with the given radius. The first node is at the "center" of the
	// circle because it's equidistant from all the other nodes, but we place it at
	// double the radius, so it should show up above all the other nodes in height.
	fn generate_circle(n: usize, radius: f64) -> Vec<Vec<f64>> {
		let mut truth: Vec<Vec<f64>> = Vec::new();
		for _ in 0..n {
			truth.push(vec![0.0; n]);
		}
		

		for i in 0..n {
			for j in i+1..n {
				let rtt: f64;
				if i == 0 {
					rtt = 2.0 * radius
				} else {
					let t1 = 2.0 * f64::consts::PI * (i as f64) / (n as f64);
					let x1 = t1.cos();
					let y1 = t1.sin();
					
					let t2 = 2.0 * f64::consts::PI * (j as f64) / (n as f64);
					let x2 = t2.cos();
					let y2 = t2.sin();

					let dx = x2 - x1;
					let dy = y2 - y1;
					let dist = (dx*dx + dy*dy).sqrt();

					rtt = dist * radius;
				}
				truth[i][j] = rtt;
				truth[j][i] = rtt;
			}
		}
		truth
	}

	// generate_Random returns a truth matrix for a set of nodes with normally
	// distributed delays, with the given mean and deviation. The RNG is re-seeded
	// so you always get the same matrix for a given size.
	fn generate_random(n: usize, mean: f64, deviation: f64) -> Vec<Vec<f64>> {
		// rand.Seed(1)

		let mut truth: Vec<Vec<f64>> = Vec::new();
		for _ in 0..n {
			truth.push(vec![0.0; n]);
		}

		for i in 0..n {
			for j in 0..n {
				let normal = Normal::new(mean, deviation).unwrap();
				let rtt = normal.sample(&mut rand::thread_rng());

				truth[i][j] = rtt;
				truth[j][i] = rtt;
			}
		}
		truth
	}

	// generate_grid returns a truth matrix as if all the nodes are in a two dimensional
	// grid with the given spacing between them.
	fn generate_grid(n: usize, spacing: f64) -> Vec<Vec<f64>> {
		let mut truth: Vec<Vec<f64>> = Vec::new();
		for _ in 0..n {
			truth.push(vec![0.0; n]);
		}

		let nsq = (n as f64).sqrt();
		for i in 0..n {
			for j in i+1..n {
				let x1 = (i as f64) % nsq;
				let y1 = (i as f64) / nsq;

				let x2 = (j as f64) % nsq;
				let y2 = (j as f64) / nsq;

				let dx = x2-x1;
				let dy = y2-y1;
		
				let dist = (dx*dx + dy*dy).sqrt();
				let rtt = dist * spacing;

				truth[i][j] = rtt;
				truth[j][i] = rtt;
			}
		}
		truth
	}

	// generate_Split returns a truth matrix as if half the nodes are close together in
	// one location and half the nodes are close together in another. The lan factor
	// is used to separate the nodes locally and the wan factor represents the split
	// between the two sides.
	fn generate_split(n: usize, lan: f64, wan: f64) -> Vec<Vec<f64>> {
		let mut truth: Vec<Vec<f64>> = Vec::new();
		for _ in 0..n {
			truth.push(vec![0.0; n]);
		}

		let split = n / 2;
		for i in 0..n {
			for j in 0..n {
				let mut rtt = lan;
				if (i <= split && j > split) || (i > split && j <= split) {
					rtt += wan
				}
				truth[i][j] = rtt;
				truth[j][i] = rtt;
			}
		}
		truth
	}

	#[test]
	fn test_performance_line() {
		const SPACING: f64 = 0.010; // 10ms
		const N: usize = 10;
		const CYCLES: isize = 1000;

		let config = Config::default();
		let mut clients = generate_clients(N, config);
		
		let truth = generate_line(N, SPACING);
		simulate(&mut clients, &truth, CYCLES);
		let stats = evaluate(clients, &truth);
		
		assert!(stats.error_avg > 0.0018 || stats.error_max > 0.0092, format!(
			"performance stats are out of spec: {:?}", stats
		));
	}

	#[test]
	fn test_performance_grid() {
		const SPACING: f64 = 0.010; // 10ms
		const N: usize = 25;
		const CYCLES: isize = 1000;

		let config = Config::default();
		let mut clients = generate_clients(N, config);

		let truth = generate_grid(N, SPACING);
		simulate(&mut clients, &truth, CYCLES);
		let stats = evaluate(clients, &truth);
		
		assert!(stats.error_avg > 0.0015 || stats.error_max > 0.022, format!(
			"performance stats are out of spec: {:?}", stats
		));
	}
	
	#[test]
	fn test_performance_random() {
		const MEAN: f64 = 0.100; // 100ms
		const DEVIATION: f64 = 0.010; // 10ms
		const N: usize = 25;
		const CYCLES: isize = 1000;

		let config = Config::default();
		let mut clients = generate_clients(N, config);

		let truth = generate_random(N, MEAN, DEVIATION);
		simulate(&mut clients, &truth, CYCLES);
		let stats = evaluate(clients, &truth);

		assert!(stats.error_avg > 0.075 || stats.error_max > 0.33, format!(
			"performance stats are out of spec: {:?}", stats
		));
	}

	#[test]
	fn test_performance_split() {
		const LAN: f64 = 0.001; // 1ms
		const WAN: f64 = 0.010; // 10ms
		const N: usize = 25;
		const CYCLES: isize = 1000;

		let config = Config::default();
		let mut clients = generate_clients(N, config);

		let truth = generate_split(N, LAN, WAN);
		simulate(&mut clients, &truth, CYCLES);
		let stats = evaluate(clients, &truth);

		assert!(stats.error_avg > 0.000060 || stats.error_max > 0.00048, format!(
			"performance stats are out of spec: {:?}", stats
		));
	}

	#[test]
	fn test_performance_height() {
		const RADIUS: f64 = 0.100; // 100ms
		const N: usize = 25; 
		const CYCLES: isize = 1000;
		const DIMENSIONALITY: usize = 2;

		// Constrain us to two dimensions so that we can just exactly represent
		// the circle.
		let mut config = Config::default();
		config.dimensionality = DIMENSIONALITY;
		let mut clients = generate_clients(N, config);
	
		// Generate truth where the first coordinate is in the "middle" because
		// it's equidistant from all the nodes, but it will have an extra radius
		// added to the distance, so it should come out above all the others.
		let truth = generate_circle(N, RADIUS);
		simulate(&mut clients, &truth, CYCLES);
	
		// Make sure the height looks reasonable with the regular nodes all in a
		// plane, and the center node up above.
		for i in 0..N {
			let coord = clients[i].get_coordinate();
			if i == 0 {
				assert!(coord.height >= 0.97*RADIUS, format!(
					"height is out of spec: {:.15}", coord.height
				));
			} else {
				assert!(coord.height <= 0.03*RADIUS,
					format!("height is out of spec: {:.15}", coord.height));
			}
		}

		let stats = evaluate(clients, &truth);
		assert!(stats.error_avg > 0.0025 || stats.error_max > 0.064,
			format!("performance stats are out of spec: {:?}", stats));
	}

	// used by test_performance_drift
	fn calc_center_error(clients: &Vec<Client>, d: usize) -> f64 {
		let mut min = clients[0].get_coordinate();
		let mut max = clients[0].get_coordinate();

		for i in 1..clients.len() {
			let coord = clients[i].get_coordinate();
			for (j, v) in coord.vec.iter().enumerate() {
				min.vec[j] = min.vec[j].min(*v);
				max.vec[j] = max.vec[j].max(*v);
			}
		}

		let mut mid = vec![0.0; d];
		// mid := make([]float64, config.Dimensionality)
		for i in 0..d {
			mid[i] = min.vec[i] + (max.vec[i]-min.vec[i])/2.0;
		}
		
		magnitude(&mid)
	}

	// used by test_performance_drift
	fn set_client_coord_vec(client: &mut Client, v: Vec<f64>) {
			let mut c = client.get_coordinate();
			c.vec = v;
			client.set_coordinate(c).unwrap();
	}

	#[test]
	fn test_performance_drift() {
		const DIST: f64 = 0.5; // 5ms
		const N: usize = 4;
		const DIMENSIONALITY: usize = 2;
		
		let mut config = Config::default();
		config.dimensionality = DIMENSIONALITY;
		let mut clients = generate_clients(N, config);
		

		// Do some icky surgery on the clients to put them into a square, up in
		// the first quadrant.
		set_client_coord_vec(&mut clients[0], vec![0.0, 0.0]);
		set_client_coord_vec(&mut clients[1], vec![0.0, DIST]);
		set_client_coord_vec(&mut clients[2], vec![DIST, DIST]);
		set_client_coord_vec(&mut clients[3], vec![DIST, DIST]);

		// Make a corresponding truth matrix. The nodes are laid out like this
		// so the distances are all equal, except for the diagonal:
		//
		// (1)  <- dist ->  (2)
		//
		//  | <- dist        |
		//  |                |
		//  |        dist -> |
		//
		// (0)  <- dist ->  (3)
		//
		let mut truth: Vec<Vec<f64>> = Vec::new();
		for _ in 0..N {
			truth.push(vec![0.0; N]);
		}

		for i in 0..N {
			for j in i + 1..N {
				let rtt: f64;
				if (i%2 == 0) && (j%2 == 0) {
					rtt = f64::consts::SQRT_2 * DIST;
				} else {
					rtt = DIST;
				}
				truth[i][j] = rtt;
				truth[j][i] = rtt;
			}
		}

		// Let the simulation run for a while to stabilize, then snap a baseline
		// for the center error.
		simulate(&mut clients, &truth, 1000);
		let baseline = calc_center_error(&clients, DIMENSIONALITY);

		// Now run for a bunch more cycles and see if gravity pulls the center
		// in the right direction.
		simulate(&mut clients, &truth, 10000);
		let error = calc_center_error(&clients, DIMENSIONALITY);
		assert!(error <= 0.8*baseline, format!(
			"drift performance out of spec: {:.15} -> {:.15}", baseline, error
		));
		// if error := calcCenterError(); error > 0.8*baseline {
		// 	t.Fatalf("drift performance out of spec: %9.6f -> %9.6f", baseline, error)
		// }
	}
}