# Image-Compression-SVD

<div align="center">
Image Compression Using Low-Rank Matrix Approximation
</div>

## Introduction

This repository is based on a class project I completed in Professor [Amir Ali Ahmadi](https://aaa.princeton.edu)'s course on computational and convex optimization (ORF363, 2021 Fall). The data compression algorithm I implemented reduces the size of a computer image by approximating the original color matrix as the product of three matrices that are much smaller.

## Theory

Let $A$ be an $m \times n$ matrix of rank $r$. After singular value decomposition, we have $$A=U \Sigma V^T,$$ where the dimensions of $U, \Sigma,$ and $V$ are respectively $m \times m$, $m \times n$ and $n \times n$. Specifically, $\Sigma$ is a matrix whose upper left $r \times r$ block is a diagonal matrix with $r$ positive scalars $\sigma_1, \sigma_2, \dots, \sigma_r$, and zero everywhere else. By convention, we may let $\sigma_1 \geq \sigma_2 \geq \dots \sigma_r$.

For any natural number $k \leq \min\{m,n\}$, let $$A_{(k)} = U_{(k)} \Sigma_{(k)} V_{(k)}^T.$$ Here, $U_{(k)}$ are the submatrix of $U$ containing its first $k$ columns, $V_{(k)}$ are the submatrix of $V$ containing its first $k$ columns, whereas $\Sigma_{(k)}$ is the upper left $k \times k$ submatrix of $\Sigma$. As a result of the [Eckart-Young-Mirsky Theorem](https://en.wikipedia.org/wiki/Low-rank_approximation), $A_{(k)}$ is an optimal solution of the minimization problem $$\min_{C \in \mathbb{R}^{m \times n}, \; rank(C) \leq k} ||A - C||_F.$$

Here, $||\cdot||_F$ is the Frobenius norm of a matrix, defined as $||X||_F = \sqrt{\sum_{i,j} X^2_{i,j}}$. An interpretation for minimizing the Frobenius norm is that we want to find an approximate matrix whose elementwise residuals with regard to $A$ has the lowest L2 norm; in other words, if we were to flatten the matrices $C$ and $A$ into vectors, we want their Euclidean distance to be as low as possible. Furthermore, in choosing the approximate matrix $C$, we want its rank to be lower than that of $A$, since the low-rank approximation would proivde a much more compact representation of the information stored in $A$.

## Usage
My Pythonic implementation of the low-rank matrix approximation solves the minimization problem stated above. Based on the compression ratio and image file provided by the user, it calculates the rank $k$ of the approximate matrix that would achieve the data compression ratio. The compression program returns three pickled numpy arrays, and displays the compressed image so that the user can determine if the image quality is acceptable.

The following is a simple Linux shell script to call the image compression applet:
```bash
cd [pathname to this directory]
python compress.py --ratio [desired compression ratio] --fname [pathname to image file]
python compress.py --ratio [desired compression ratio] --fname [pathname to image file] --as_gray # compress a grayscale image
```