#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]

use core::f32;

use npyz::NpyFile;
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
/// *   sigle_data_len: length of a single piece of data in the data
/// :return
/// *   centroids: indices of data points to be used as centroids
fn initialize_cluster(data: &Vec<f32>, k: usize, sigle_data_len: usize) -> Vec<usize> {
    let mut data_idx: Vec<usize> = (0..data.len() / sigle_data_len).collect();
    for c in 1..k {
        print!("\rCentroids initiated {}", c);
        let next_centroid_idx = data_idx[c - 1..]
            .iter()
            .map(|d| {
                data_idx[..c].iter().fold(f32::INFINITY, |a, &b| {
                    a.min(dist_calc(
                        &data[*d * sigle_data_len..((*d + 1) * sigle_data_len)],
                        &data[b * sigle_data_len..((b + 1) * sigle_data_len)],
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

/// Calculate the meand 1d vec (centroid) form a vec containing data of multiple data points
/// :parameter
/// *   `data`: the vectors for which the mean vector should be calculated
/// *   `idxs`: the indices of the vectors which should be used to calculate the mean
/// *   sigle_data_len: length of a single piece of data in the data
/// :return
/// *   `mean_by_idx`: mean vec of all indexed vectors
fn mean_by_idx(data: &[f32], idxs: &Vec<usize>, sigle_data_len: usize) -> Vec<f32> {
    let n_samples = idxs.len() as f32;
    idxs.iter().fold(vec![0.0; sigle_data_len], |a, b| {
        a.iter()
            .enumerate()
            .map(|(cx, x)| x + data[b * sigle_data_len..((b + 1) * sigle_data_len)][cx] / n_samples)
            .collect()
    })
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
    sigle_data_len: usize,
    iterations: usize,
    n_cluster: usize,
    tol: f32,
) {
    let mut best_inertia = f32::INFINITY;
    for iteration in 0..iterations {
        print!("\rStarting iteration {}", iteration);
        // find the closest centroid for each data point
        cluster_asign.iter_mut().enumerate().for_each(|(cx, x)| {
            centroids
                .chunks_exact(sigle_data_len)
                .map(|y| dist_calc(&data[cx * sigle_data_len..((cx + 1) * sigle_data_len)], y))
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| *x = idx)
                .unwrap()
        });

        for cn in 0..n_cluster {
            // number of members of the cluster
            let n_cn = cluster_asign.iter().filter(|&x| *x == cn).count();
            // calculate new centroid of assigned points to cluster cn
            let cn_cent = cluster_asign
                .iter()
                .enumerate()
                .filter(|(_, &x)| x == cn)
                .fold(vec![0.0; sigle_data_len], |a, b| {
                    a.iter()
                        .enumerate()
                        .map(|(cy, y)| {
                            y + data[b.0 * sigle_data_len..((b.0 + 1) * sigle_data_len)][cy]
                                / (n_cn as f32)
                        })
                        .collect()
                });
            centroids[cn * sigle_data_len..((cn + 1) * sigle_data_len)]
                .copy_from_slice(&cn_cent[..]);
        }
        // the sum of euclidean distances to all points to their cluster centers
        let interia = cluster_asign.iter().enumerate().fold(0.0, |sum, (cx, x)| {
            sum + dist_calc(
                &data[cx * sigle_data_len..((cx + 1) * sigle_data_len)],
                &data[x * sigle_data_len..((x + 1) * sigle_data_len)],
            )
            .abs()
        });
        // how much the inertia changed from last iteration
        let delta_inertia = best_inertia - interia;
        if delta_inertia.abs() <= tol {
            println!("\nDelta inertia tolerance reached - stopping");
            break;
        }
        if delta_inertia > 0.0 {
            best_inertia = interia;
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

    let single_data_size = 16;
    let n_cluster = 400;

    let centroids = initialize_cluster(&sample_data, n_cluster, single_data_size);
    /*
    let file = File::create("cluster_centroids_rs.txt").unwrap();
    let mut file = BufWriter::new(file);
    for v in &centroids {
        writeln!(file, "{}", v).unwrap();
    }
    */
    let mut centroids: Vec<f32> = centroids
        .iter()
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
