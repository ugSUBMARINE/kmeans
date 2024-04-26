#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]

use npyz::NpyFile;
use rayon::prelude::*;
use std::borrow::Borrow;
use std::fs::File;
use std::io::{BufWriter, Write};
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
fn get_matrices(base_file_path: String, side_len: i32) -> Vec<f32> {
    let total_mat_size = side_len * side_len;
    let mut triu_idx: Vec<usize> = vec![];
    for i in 0..side_len {
        for j in i + 1..side_len {
            triu_idx.push((side_len * i + j) as usize)
        }
    }
    let mut all_triu: Vec<f32> = vec![];

    for path in WalkDir::new(&base_file_path)
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
        let bytes = std::fs::read(path.path()).unwrap();
        let npy = NpyFile::new(&bytes[..]).unwrap().into_vec::<f32>().unwrap();
        let npyoi_pre = npy.chunks_exact(total_mat_size as usize);
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
    }
    all_triu
}

/// Get indices of the data to create initial clusters for k-means
/// :parameter
/// *   data: the data to be clustered
/// *   k: the number of clusters to be created
/// *   single_data_len: length of a single piece of data in the data
/// :return
/// *   centroids: indices of data points to be used as centroids
// fn initialize_cluster(data: &Vec<f32>, k: usize, single_data_len: usize) -> (Vec<usize>, Vec<f32>) {
fn initialize_cluster(data: &Vec<f32>, k: usize, single_data_len: usize) -> Vec<usize> {
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
    println!("");
    data_idx[..k].to_vec()
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
                    .map(|y| *y / &cluster_nums[cx])
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<f32>>();
        let center_shift = par_dist_calc(&centroids, &ncn).powi(2);
        centroids.copy_from_slice(&ncn[..]);

        // the sum of euclidean distances to all points to their cluster centers
        let interia = cluster_asign
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
            println!("\nCluster shift tolerance reached {}", center_shift);
            break;
        }
    }
    println!("")
}

fn main() {
    /*
    TEST SHAPE matrices_16clean and plane.npy and their for small nr of matrices_16clean whether the triu matches
    let matrix_data = get_matrices(
        // "/media/gwirn/D/alphafold_models/ecoli/matrices_16clean/".to_string(),
        "./".to_string(),
        3,
    );
    println!("{}", matrix_data.len())
    */
    let bytes = std::fs::read("./test_data.npy").unwrap();
    let sample_data = NpyFile::new(&bytes[..]).unwrap().into_vec::<f32>().unwrap();

    let single_data_size = 2;
    let n_cluster = 4;

    let centroids = initialize_cluster(&sample_data, n_cluster, single_data_size);
    let file = File::create("initial_centroids_rs.txt").unwrap();
    let mut file = BufWriter::new(file);
    for v in &centroids {
        writeln!(file, "{}", v).unwrap();
    }
    let mut centroids: Vec<f32> = centroids
        .par_iter()
        .flat_map(|x| sample_data[x * single_data_size..((x + 1) * single_data_size)].to_vec())
        .collect();
    let mut cluster_asign = vec![0; sample_data.len() / single_data_size];
    kmeans(
        &sample_data,
        &mut centroids,
        &mut cluster_asign,
        single_data_size,
        200,
        n_cluster,
        1e-4,
    );
    // println!("{:?}", cluster_asign);
    let file = File::create("cluster_rs.txt").unwrap();
    let mut file = BufWriter::new(file);
    for v in cluster_asign {
        writeln!(file, "{}", v).unwrap();
    }
}
