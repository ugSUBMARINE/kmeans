# Kmeans for Numpy Distance Matrices

This repository implements the K-means clustering algorithm for numpy distance matrices, written in Rust.

## Description

This project provides an efficient implementation of the K-means algorithm, optimized for processing distance matrices stored in numpy format (`.npy`). It includes various initialization schemes and optimizations for better performance and accuracy.

## Installation

To use this project, you will need to have Rust installed. You can install Rust from the official [Rust website](https://www.rust-lang.org/).

To get started, clone the repository and build the project using Cargo:

```sh
git clone https://github.com/ugSUBMARINE/kmeans.git
cd kmeans
cargo build --release
```

## Usage

To run the K-means algorithm on your data, use the following command:

```sh
cargo run --release -- <path_to_distance_matrix.npy> <number_of_clusters>
```

Replace `<path_to_distance_matrix.npy>` with the path to your numpy distance matrix file and `<number_of_clusters>` with the desired number of clusters.

## Features

- Efficient K-means clustering for numpy distance matrices.
- Various initialization schemes including Hartigan's algorithm.
- Optimized for performance with parallel processing capabilities.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
