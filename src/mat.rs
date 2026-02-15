use num_traits::Float;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat<const N: usize, const M: usize, T: Float = f64> {
    pub data: [[T; M]; N],
}

impl<const N: usize, const M: usize, T: Float> Mat<N, M, T> {
    pub fn from_data(data: [[T; M]; N]) -> Self {
        Self { data }
    }

    pub fn zeros() -> Self {
        Self {
            data: [[T::zero(); M]; N],
        }
    }

    pub fn set_block<const P: usize, const Q: usize>(
        &mut self,
        row: usize,
        col: usize,
        block: &Mat<P, Q, T>,
    ) {
        for i in 0..P {
            for j in 0..Q {
                self.data[row + i][col + j] = block.data[i][j];
            }
        }
    }

    pub fn transpose(&self) -> Mat<M, N, T> {
        Mat {
            data: std::array::from_fn(|j| std::array::from_fn(|i| self.data[i][j])),
        }
    }
}

// Row vector: Mat<1, M>
impl<const M: usize, T: Float> Mat<1, M, T> {
    pub fn row(data: [T; M]) -> Self {
        Self { data: [data] }
    }
}

// Column vector: Mat<N, 1>
impl<const N: usize, T: Float> Mat<N, 1, T> {
    pub fn col(data: [T; N]) -> Self {
        Self {
            data: std::array::from_fn(|i| [data[i]]),
        }
    }
}

// Square matrix methods
impl<const N: usize, T: Float> Mat<N, N, T> {
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..N {
            m.data[i][i] = T::one();
        }
        m
    }
}

// Mat<N, K> * Mat<K, M> -> Mat<N, M>
impl<const N: usize, const K: usize, const M: usize, T: Float> Mul<Mat<K, M, T>> for Mat<N, K, T> {
    type Output = Mat<N, M, T>;
    fn mul(self, rhs: Mat<K, M, T>) -> Mat<N, M, T> {
        Mat {
            data: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let mut sum = T::zero();
                    for k in 0..K {
                        sum = sum + self.data[i][k] * rhs.data[k][j];
                    }
                    sum
                })
            }),
        }
    }
}

// Mat<N, M> * [T; M] -> [T; N]
impl<const N: usize, const M: usize, T: Float> Mul<[T; M]> for Mat<N, M, T> {
    type Output = [T; N];
    fn mul(self, rhs: [T; M]) -> [T; N] {
        std::array::from_fn(|i| {
            let mut sum = T::zero();
            for j in 0..M {
                sum = sum + self.data[i][j] * rhs[j];
            }
            sum
        })
    }
}

// Mat + Mat
impl<const N: usize, const M: usize, T: Float> Add for Mat<N, M, T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            data: std::array::from_fn(|i| {
                std::array::from_fn(|j| self.data[i][j] + rhs.data[i][j])
            }),
        }
    }
}

// Mat - Mat
impl<const N: usize, const M: usize, T: Float> Sub for Mat<N, M, T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            data: std::array::from_fn(|i| {
                std::array::from_fn(|j| self.data[i][j] - rhs.data[i][j])
            }),
        }
    }
}

// -Mat
impl<const N: usize, const M: usize, T: Float> Neg for Mat<N, M, T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| -self.data[i][j])),
        }
    }
}

// Mat * scalar
impl<const N: usize, const M: usize, T: Float> Mul<T> for Mat<N, M, T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| self.data[i][j] * rhs)),
        }
    }
}

// f64 * Mat
impl<const N: usize, const M: usize> Mul<Mat<N, M, f64>> for f64 {
    type Output = Mat<N, M, f64>;
    fn mul(self, rhs: Mat<N, M, f64>) -> Mat<N, M, f64> {
        rhs * self
    }
}

// f32 * Mat
impl<const N: usize, const M: usize> Mul<Mat<N, M, f32>> for f32 {
    type Output = Mat<N, M, f32>;
    fn mul(self, rhs: Mat<N, M, f32>) -> Mat<N, M, f32> {
        rhs * self
    }
}

// Index by row: m[i] returns &[T; M], so m[i][j] works naturally
impl<const N: usize, const M: usize, T: Float> Index<usize> for Mat<N, M, T> {
    type Output = [T; M];
    fn index(&self, i: usize) -> &[T; M] {
        &self.data[i]
    }
}

impl<const N: usize, const M: usize, T: Float> IndexMut<usize> for Mat<N, M, T> {
    fn index_mut(&mut self, i: usize) -> &mut [T; M] {
        &mut self.data[i]
    }
}

// Type aliases
pub type Mat3<T = f64> = Mat<3, 3, T>;

impl<T: Float> Mat3<T> {
    pub fn skew(v: [T; 3]) -> Self {
        let z = T::zero();
        Self::from_data([[z, -v[2], v[1]], [v[2], z, -v[0]], [-v[1], v[0], z]])
    }
}
pub type Mat4<T = f64> = Mat<4, 4, T>;
pub type Mat6<T = f64> = Mat<6, 6, T>;
