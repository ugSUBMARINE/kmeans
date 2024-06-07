#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![allow(unused_imports)]

use clap::Parser;
use npyz::NpyFile;
use rayon::prelude::*;
use std::collections::HashSet;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::usize;
use walkdir::WalkDir;

/// Calculate the euclidean distance between two 1D vec
/// :parameter
/// *   `arr1`, `arr2`: vec between which the distance should be calculated
/// :return
/// * `distance` between the vectors
fn dist_calc(arr1: &[f32], arr2: &[f32]) -> f32 {
    arr1.iter()
        .zip(arr2)
        .fold(0., |sum, (a1, a2)| sum + (f32::powi(a1 - a2, 2)))
        .sqrt()
}
/// Calculate the euclidean distance between two 1D vec in parallel (only for big vec)
/// :parameter
/// *   `arr1`, `arr2`: vec between which the distance should be calculated
/// :return
/// * `distance` between the vectors
fn par_dist_calc(arr1: &[f32], arr2: &[f32]) -> f32 {
    arr1.par_iter()
        .zip(arr2)
        .fold(|| 0.0, |sum, (x, y)| sum + (x - y).powi(2))
        .reduce(|| 0.0, |a: f32, b: f32| a + b)
        .sqrt()
}

/// Read npy files containing square distance matrices and extract their upper triangle as 1D vec
/// :parameter
/// *   `base_file_path`: root file path of the directory containing the npy files
/// *   `side_len`: length of the distance matrix
/// :return
/// *   `all_triu`: 1D vec containing all the upper triangle
fn get_matrices(
    base_file_path: String,
    side_len: usize,
) -> Result<(Vec<f32>, Vec<String>), std::io::Error> {
    let total_mat_size = side_len * side_len;
    let mut triu_idx: Vec<usize> = vec![];
    for i in 0..side_len {
        for j in i + 1..side_len {
            triu_idx.push(side_len * i + j)
        }
    }
    let mut all_triu: Vec<f32> = vec![];
    let mut all_paths: Vec<String> = vec![];

    let mut path_count = 1;
    for path in WalkDir::new(base_file_path)
        .into_iter()
        .filter(|s| {
            s.as_ref()
                .unwrap()
                .path()
                .to_str()
                .unwrap()
                .ends_with(".npy")
        })
        .map(|e| e.unwrap())
    {
        all_paths.push(path.path().to_str().unwrap().to_string());
        let bytes = std::fs::read(path.path())?;
        let npy: Vec<f32> = NpyFile::new(&bytes[..])?
            .into_vec::<f64>()?
            .iter()
            .map(|x| *x as f32)
            .collect();
        let npyoi_pre = npy.chunks_exact(total_mat_size);
        if !npyoi_pre.remainder().is_empty() {
            panic!(
                "npy data size is not compatible with given side_len {}",
                side_len
            )
        }
        let mut npyoi = npyoi_pre
            .flat_map(|x| {
                x.iter()
                    .enumerate()
                    .filter(|&(c, _)| triu_idx.contains(&c))
                    .map(|(_, i)| *i)
            })
            .collect::<Vec<f32>>();
        all_triu.append(&mut npyoi);
        print!("\rMatrix files read {}", path_count);
        path_count += 1;
    }
    println!();
    Ok((all_triu, all_paths))
}

/// Get indices of the data to create initial clusters for k-means via kmeans++
/// :parameter
/// *   data: the data to be clustered
/// *   k: the number of clusters to be created
/// *   single_data_len: length of a single piece of data in the data
/// :return
/// *   centroids: coordinates of data points to be used as centroids
fn maxmin_init(data: &Vec<f32>, k: usize, single_data_len: usize) -> Vec<f32> {
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
fn naive_sharding_init(data: &[f32], k: usize, single_data_len: usize) -> Vec<f32> {
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
///     N dataspoints based on their indices with (1 +  (i − 1)N/K)
/// :parameter
/// *   data: the data to be clustered
/// *   k: the number of clusters to be created
/// *   single_data_len: length of a single piece of data in the data
/// :return
/// *   centroids: coordinates of data points to be used as centroids
fn hartigan_init(data: &[f32], k: usize, single_data_len: usize) -> Vec<f32> {
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
/*
Initialize k means with random values
--> For a given number of iterations:
    --> Iterate through items:
        --> Find the mean closest to the item by calculating
        the euclidean distance of the item with each of the means
        --> Assign item to mean
        --> Update mean by shifting it to the average of the items in that cluster*/
fn kmeans(
    data: &[f32],
    centroids: &mut [f32],
    cluster_asign: &mut [usize],
    single_data_len: usize,
    iterations: usize,
    n_cluster: usize,
    tol: f32,
) {
    for iteration in 0..iterations {
        print!("\rStarting iteration {}", iteration);
        // find the closest centroid for each data point
        cluster_asign
            .into_par_iter()
            .enumerate()
            .for_each(|(cx, x)| {
                centroids
                    .chunks_exact(single_data_len)
                    .map(|y| {
                        dist_calc(&data[cx * single_data_len..((cx + 1) * single_data_len)], y)
                    })
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(idx, _)| *x = idx)
                    .unwrap()
            });

        let mut ncn = vec![0.; centroids.len()];
        // count how many items are in one cluster to later calc the mean
        let mut cluster_nums = vec![0.; n_cluster];
        for (ci, i) in cluster_asign.iter().enumerate() {
            cluster_nums[*i] += 1.;
            let ncn_slice = &ncn[*i * single_data_len..((*i + 1) * single_data_len)]
                .iter()
                .zip(&data[ci * single_data_len..((ci + 1) * single_data_len)])
                .map(|(x, y)| x + y)
                .collect::<Vec<f32>>();
            ncn[*i * single_data_len..((*i + 1) * single_data_len)].copy_from_slice(&ncn_slice[..]);
        }
        let ncn = ncn
            .par_chunks_mut(single_data_len)
            .enumerate()
            .flat_map(|(cx, x)| {
                x.iter_mut()
                    .map(|y| *y / cluster_nums[cx])
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<f32>>();
        let center_shift = par_dist_calc(centroids, &ncn).powi(2);
        centroids.copy_from_slice(&ncn[..]);

        // the sum of euclidean distances to all points to their cluster centers
        let inertia = cluster_asign
            .par_iter()
            .enumerate()
            .fold(
                || 0.0,
                |sum, (cx, x)| {
                    sum + dist_calc(
                        &data[cx * single_data_len..((cx + 1) * single_data_len)],
                        &centroids[x * single_data_len..((x + 1) * single_data_len)],
                    )
                    .abs()
                },
            )
            .reduce(|| 0.0, |a: f32, b: f32| a + b);
        // how much the inertia changed from last iteration
        if center_shift <= tol {
            println!(
                "\nCluster shift tolerance reached {} with inertia of {}",
                center_shift, inertia
            );
            break;
        }
        if iteration == iterations - 1 {
            println!(
                "\nClusterind iteration max reached with inertia of {}",
                inertia
            );
        }
    }
}

/// find the member of the cluster with the minimum distance to all other members
/// therefore the most representative (most similar to all other ones) of the cluster
/// :parameter
/// *   data: the data that represents all members of the cluster
/// *   single_data_len: length of a single piece of data in the data
/// :return
/// *   the index of the member with the closest distance to all others
fn find_representative(
    data: Vec<&[f32]>,
    cluster: &[usize],
    coi: usize,
    single_data_len: usize,
) -> Option<usize> {
    data.par_iter()
        .map(|x| data.iter().map(|y| dist_calc(x, y)).sum::<f32>())
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

/// KMeans clustering for np distance matrices
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// file path to the npy file stored matrices
    #[arg(short, long)]
    inpath: String,

    /// file path where the cluster results should be stored
    #[arg(short, long)]
    outpath: String,

    /// Side length of the distance matrices
    #[arg(short, long)]
    side_len: usize,

    /// number of clusters to build
    #[arg(short, long)]
    n_cluster: usize,

    /// initialization method
    #[arg(short, long, default_value = "minmax")]
    m_init: String,
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.outpath).unwrap();
    let (data, fpaths) = get_matrices(args.inpath, args.side_len).unwrap();
    let file = File::create(Path::new(&args.outpath).join("fpaths.txt")).unwrap();
    let mut file = BufWriter::new(file);
    for v in fpaths {
        writeln!(file, "{:?}", v).unwrap()
    }

    let single_data_size = args.side_len;
    let n_cluster = args.side_len;

    let mut centroids = match args.m_init.as_str() {
        "minmax" => maxmin_init(&data, n_cluster, single_data_size),
        "sharding" => naive_sharding_init(&data, n_cluster, single_data_size),
        "hart" => hartigan_init(&data, n_cluster, single_data_size),
        _ => panic!("Invalid initialization method '{}'", args.m_init),
    };
    let file = File::create(Path::new(&args.outpath).join("initial_centroids_rs.txt")).unwrap();
    let mut file = BufWriter::new(file);
    for v in centroids.chunks_exact(single_data_size) {
        writeln!(file, "{:?}", v).unwrap();
    }
    let mut cluster_asign = vec![0; data.len() / single_data_size];
    kmeans(
        &data,
        &mut centroids,
        &mut cluster_asign,
        single_data_size,
        200,
        n_cluster,
        1e-4,
    );
    let file = File::create(Path::new(&args.outpath).join("cluster_rs.txt")).unwrap();
    let mut file = BufWriter::new(file);
    for v in &cluster_asign {
        writeln!(file, "{}", v).unwrap();
    }
    let mut cluster_rep_idx: Vec<usize> = Vec::with_capacity(n_cluster);
    for i in 0..n_cluster {
        print!("\rRepresentative search cluster {}", i);
        let (tsidx, test_sup): (Vec<_>, Vec<_>) = data
            .par_chunks_exact(single_data_size)
            .zip(cluster_asign.par_iter())
            .enumerate()
            .filter(|(_, (_, &c))| c == i)
            .map(|(cx, (x, &c))| (cx, x))
            .unzip();
        match find_representative(test_sup, &cluster_asign, i, single_data_size) {
            Some(x) => {
                cluster_rep_idx.push(tsidx[x]);
            }
            None => {
                println!("No cluster representative for {}", i);
                continue;
            }
        };
    }
    println!();
    let file = File::create(Path::new(&args.outpath).join("representatives.txt")).unwrap();
    let mut file = BufWriter::new(file);
    for v in cluster_rep_idx {
        writeln!(file, "{:?}", v).unwrap()
    }
}
