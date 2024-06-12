use crate::find_representative;
use crate::init_shemes::{
    grid_init, hartigan_init, maxmin_init, naive_sharding_init, simple_cluster_seek,
};
use crate::kmeans;
use npyz::NpyFile;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::usize;
use walkdir::WalkDir;
/// Calculate the euclidean distance between two 1D vec
/// :parameter
/// *   `arr1`, `arr2`: vec between which the distance should be calculated
/// :return
/// * `distance` between the vectors
pub fn dist_calc(arr1: &[f32], arr2: &[f32]) -> f32 {
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
pub fn par_dist_calc(arr1: &[f32], arr2: &[f32]) -> f32 {
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
pub fn get_matrices(
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
pub fn run_test() {
    let (data, fpaths) = get_matrices("./test_data/".to_string(), 4).unwrap();
    let file = File::create("fpaths.txt").unwrap();
    let mut file = BufWriter::new(file);
    for v in fpaths {
        writeln!(file, "{:?}", v).unwrap()
    }
    let bytes = std::fs::read("./test_data.npy").unwrap();
    let data = NpyFile::new(&bytes[..]).unwrap().into_vec::<f32>().unwrap();

    let single_data_size = 2;
    let n_cluster = 1000;

    let mut centroids = maxmin_init(&data, n_cluster, single_data_size);
    let mut centroids = simple_cluster_seek(&data, n_cluster, single_data_size);
    let mut centroids = naive_sharding_init(&data, n_cluster, single_data_size);
    let mut centroids = hartigan_init(&data, n_cluster, single_data_size);
    let mut centroids = grid_init(&data, n_cluster, single_data_size);
    let file = File::create("initial_centroids_rs.txt").unwrap();
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
    let file = File::create("cluster_rs.txt").unwrap();
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
    let file = File::create("representatives.txt").unwrap();
    let mut file = BufWriter::new(file);
    for v in cluster_rep_idx {
        writeln!(file, "{:?}", v).unwrap()
    }
}
