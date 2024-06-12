#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::init_shemes::{grid_init, hartigan_init, maxmin_init, naive_sharding_init};
use crate::utils::{dist_calc, get_matrices, par_dist_calc, run_test};
use clap::Parser;
use rayon::prelude::*;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::usize;
mod init_shemes;
mod utils;
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
fn find_representative(data: Vec<&[f32]>) -> Option<usize> {
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
    run_test();
    let args = Args::parse();
    fs::create_dir_all(&args.outpath).unwrap();
    let (data, fpaths) = get_matrices(args.inpath, args.side_len).unwrap();
    let file = File::create(Path::new(&args.outpath).join("fpaths.txt")).unwrap();
    let mut file = BufWriter::new(file);
    for v in fpaths {
        writeln!(file, "{:?}", v).unwrap()
    }

    let single_data_size = args.side_len;
    let n_cluster = args.n_cluster;
    println!("Matrices found {}", data.len() / single_data_size);

    let mut centroids = match args.m_init.as_str() {
        "minmax" => maxmin_init(&data, n_cluster, single_data_size),
        "sharding" => naive_sharding_init(&data, n_cluster, single_data_size),
        "hart" => hartigan_init(&data, n_cluster, single_data_size),
        "grid" => grid_init(&data, n_cluster, single_data_size),
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
            .map(|(cx, (x, _))| (cx, x))
            .unzip();
        match find_representative(test_sup) {
            Some(x) => {
                cluster_rep_idx.push(tsidx[x]);
            }
            None => {
                println!("\nNo cluster representative for {}", i);
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
