use crate::utils::{dist_calc, par_dist_calc};
use rand::prelude::*;
use rayon::prelude::*;

/// Get indices of the data to create initial clusters for k-means via kmeans++
/// :parameter
/// *   data: the data to be clustered
/// *   k: the number of clusters to be created
/// *   single_data_len: length of a single piece of data in the data
/// :return
/// *   centroids: coordinates of data points to be used as centroids
pub fn maxmin_init(data: &Vec<f32>, k: usize, single_data_len: usize) -> Vec<f32> {
    let mut data_idx: Vec<usize> = (0..data.len() / single_data_len).collect();
    for c in 1..k {
        print!("\rCentroids initiated {}", c);
        let next_centroid_idx = data_idx[c - 1..]
            .par_iter()
            .map(|d| {
                data_idx[..c].iter().fold(f32::INFINITY, |a, &b| {
                    a.min(dist_calc(
                        &data[*d * single_data_len..((*d + 1) * single_data_len)],
                        &data[b * single_data_len..((b + 1) * single_data_len)],
                    ))
                })
            })
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx + (c - 1))
            .unwrap();
        // swap new centroid idx with first non centroid element in data_idx
        data_idx.swap(c, next_centroid_idx);
    }
    println!();
    let centroids: Vec<f32> = data_idx[..k]
        .par_iter()
        .flat_map(|x| data[x * single_data_len..((x + 1) * single_data_len)].to_vec())
        .collect();
    centroids
}

/// Get indices of the data to create initial clusters for k-means via naive sharding
///     Sort data according to the sum of their attributes
///     split them in K chunks and calculate the mean of these data points as a centroid
/// :parameter
/// *   data: the data to be clustered
/// *   k: the number of clusters to be created
/// *   single_data_len: length of a single piece of data in the data
/// :return
/// *   centroids: coordinates of data points to be used as centroids
pub fn naive_sharding_init(data: &[f32], k: usize, single_data_len: usize) -> Vec<f32> {
    let val_sum = data
        .chunks_exact(single_data_len)
        .map(|x| x.iter().sum::<f32>())
        .collect::<Vec<f32>>();
    let mut init_idx: Vec<usize> = (0..val_sum.len()).collect();
    // init_idx.par_sort_by(|i, j| val_sum[*i].partial_cmp(&val_sum[*j]).unwrap());
    init_idx.par_sort_by(|i, j| val_sum[*i].total_cmp(&val_sum[*j]));
    let chunk_size = (init_idx.len() as f32 / k as f32).floor() as usize;
    let mut ncent: Vec<f32> = vec![0.; k * single_data_len];
    let mut idx_chunks = init_idx.chunks_exact(chunk_size).enumerate().peekable();
    for (ci, i) in idx_chunks.by_ref() {
        print!("\rCentroids initiated {}", ci);
        if ci < k - 1 {
            let i_size = i.len() as f32;
            let i_vec = i.iter().fold(vec![0.; single_data_len], |acc, idx| {
                acc.iter()
                    .enumerate()
                    .map(|(ix, x)| x + data[idx * single_data_len + ix] / i_size)
                    .collect()
            });
            ncent[single_data_len * ci..((ci + 1) * single_data_len)].copy_from_slice(&i_vec[..]);
        } else {
            break;
        }
    }
    let n_remainders = idx_chunks.len() as f32;
    if idx_chunks.peek().is_some() {
        let mut remainder = vec![0.; single_data_len];
        for (_, i) in idx_chunks {
            let i_size = i.len() as f32;
            let i_vec = i.iter().fold(vec![0.; single_data_len], |acc, idx| {
                acc.iter()
                    .enumerate()
                    .map(|(ix, x)| x + data[idx * single_data_len + ix] / i_size)
                    .collect()
            });
            for (cn, n) in i_vec.iter().enumerate() {
                remainder[cn] += n / n_remainders
            }
        }
        ncent[(k - 1) * single_data_len..k * single_data_len].copy_from_slice(&remainder[..]);
        print!("\rCentroids initiated {}", k);
    };
    println!();
    ncent
}

/// Get indices of the data to create initial clusters for k-means via hartigan method
///     Sort all point according to their distance to the mean point and then select K cluster from
///     N data points based on their indices with (1 +  (i − 1)N/K)
/// :parameter
/// *   data: the data to be clustered
/// *   k: the number of clusters to be created
/// *   single_data_len: length of a single piece of data in the data
/// :return
/// *   centroids: coordinates of data points to be used as centroids
pub fn hartigan_init(data: &[f32], k: usize, single_data_len: usize) -> Vec<f32> {
    let n_samples = (data.len() / single_data_len) as f32;
    let kf = k as f32;
    let mean_point =
        data.chunks_exact(single_data_len)
            .fold(vec![0.; single_data_len], |acc, chunk| {
                acc.iter()
                    .enumerate()
                    .map(|(cx, x)| x + (chunk[cx] / n_samples))
                    .collect()
            });
    let dist_to_mean: Vec<f32> = data
        .par_chunks_exact(single_data_len)
        .map(|x| dist_calc(x, &mean_point))
        .collect();
    let mut dist_sort = (0..dist_to_mean.len()).collect::<Vec<_>>();
    dist_sort.sort_by(|&i, &j| dist_to_mean[i].total_cmp(&dist_to_mean[j]));

    let data_idx: Vec<usize> = (0..k)
        .into_par_iter()
        .map(|x| (1. + (x as f32 - 1.) * n_samples / kf) as usize)
        .collect();

    let centroids: Vec<f32> = data_idx
        .par_iter()
        .flat_map(|x| data[x * single_data_len..((x + 1) * single_data_len)].to_vec())
        .collect();

    centroids
}
fn grid_idx(ranges: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut res: Vec<Vec<f32>> = vec![];
    let lengths: Vec<_> = ranges.iter().map(|x| x.len()).collect();
    let mut indices = vec![0; ranges.len()];
    loop {
        print!("\rGrid points sampled: {}", res.len());
        res.push(
            ranges
                .iter()
                .enumerate()
                .map(|(cx, _)| ranges[cx][indices[cx]])
                .collect(),
        );
        let mut i = 0;
        indices[i] += 1;
        while i < ranges.len() - 1 && indices[i] == lengths[i] {
            indices[i] = 0;
            i += 1;
            indices[i] += 1
        }
        if indices[indices.len() - 1] == lengths[lengths.len() - 1] {
            break;
        }
    }
    res
}

/// Get indices of the data to create initial clusters for k-means via an even grid
///     Lay an k-dimensional grid with a spacing that results in as close as possible k grid points
///     and randomly remove grid points that are more than `k` to find initial centers
/// :parameter
/// *   data: the data to be clustered
/// *   k: the number of clusters to be created
/// *   single_data_len: length of a single piece of data in the data
/// :return
/// *   centroids: coordinates of data points to be used as centroids
pub fn grid_init(data: &[f32], k: usize, single_data_len: usize) -> Vec<f32> {
    let grid_side_len = (f32::powf(k as f32, 1. / single_data_len as f32)).ceil();
    let mut maxs: Vec<f32> = vec![f32::NEG_INFINITY; single_data_len];
    let mut mins: Vec<f32> = vec![f32::INFINITY; single_data_len];

    for i in data.chunks_exact(single_data_len) {
        for (cj, j) in maxs.iter_mut().enumerate() {
            if *j < i[cj] {
                *j = i[cj]
            }
        }
        for (cj, j) in mins.iter_mut().enumerate() {
            if *j > i[cj] {
                *j = i[cj]
            }
        }
    }

    let mut all_ranges: Vec<Vec<f32>> = Vec::new();
    for i in 0..maxs.len() {
        all_ranges.push({
            let dx = (maxs[i] - mins[i]) / (grid_side_len - 1.);
            let mut i_range = vec![mins[i]; grid_side_len as usize];
            for j in 1..(grid_side_len as usize) {
                i_range[j] = i_range[j - 1] + dx
            }
            i_range
        });
    }
    let mut grid_points = grid_idx(all_ranges);
    let mut gp_idx: Vec<_> = (0..grid_points.len()).collect();
    let mut rng = rand::thread_rng();
    gp_idx.shuffle(&mut rng);
    gp_idx = gp_idx[..k].to_vec();
    let mut r_idx = 0;
    grid_points.retain(|_| {
        r_idx += 1;
        gp_idx.contains(&(r_idx - 1))
    });
    grid_points
        .iter()
        .flat_map(|x| x.clone())
        .collect::<Vec<f32>>()
}

/// Get indices of the data to create initial clusters for k-means via simple cluster seek method
///     Use the first data point as first cluster then search through next data points until one is
///     found that is further away than `t` to the last cluster center - repeat until enough
///     center are found
///     `t` is calculated as the maximum distance between point attributes / 2.5
/// :parameter
/// *   data: the data to be clustered
/// *   k: the number of clusters to be created
/// *   single_data_len: length of a single piece of data in the data
/// :return
/// *   centroids: coordinates of data points to be used as centroids
pub fn simple_cluster_seek(data: &[f32], k: usize, single_data_len: usize) -> Vec<f32> {
    let mut cluster_center: Vec<f32> = Vec::with_capacity(k);
    let cc_max_size = k * single_data_len;
    let mut maxs: Vec<f32> = vec![f32::NEG_INFINITY; single_data_len];
    let mut mins: Vec<f32> = vec![f32::INFINITY; single_data_len];

    for i in data.chunks_exact(single_data_len) {
        for (cj, j) in maxs.iter_mut().enumerate() {
            if *j < i[cj] {
                *j = i[cj]
            }
        }
        for (cj, j) in mins.iter_mut().enumerate() {
            if *j > i[cj] {
                *j = i[cj]
            }
        }
    }
    let t = dist_calc(&maxs, &mins) / 3.5;

    let mut n_found = 0;
    for (ci, i) in data.chunks_exact(single_data_len).enumerate() {
        if ci == 0 {
            cluster_center.extend_from_slice(i)
        } else {
            let cc_len = cluster_center.len();
            if dist_calc(i, &cluster_center[(cc_len - single_data_len)..(cc_len)]) > t {
                cluster_center.extend_from_slice(i);
                n_found += 1;
            }
        }
        if n_found >= k - 1 {
            println!("Found enough cluster");
            break;
        }
    }

    cluster_center
}
