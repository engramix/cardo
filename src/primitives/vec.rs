use num_traits::Float;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VecN<T, const N: usize> {
    pub data: [T; N],
}

impl<T: Float, const N: usize> VecN<T, N> {
    pub fn new(data: [T; N]) -> Self {
        Self { data }
    }

    pub fn zero() -> Self {
        Self {
            data: [T::zero(); N],
        }
    }

    pub fn dot(&self, rhs: &Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self.data[i] * rhs.data[i];
        }
        sum
    }

    pub fn norm_squared(&self) -> T {
        self.dot(self)
    }

    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        let mut data = self.data;
        data.iter_mut().for_each(|x| *x = *x / n);
        Self { data }
    }

    pub fn coeffs(&self) -> &[T] {
        &self.data
    }
}

impl<T, const N: usize> Index<usize> for VecN<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for VecN<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

impl<T: Float, const N: usize> Add for VecN<T, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            data: std::array::from_fn(|i| self.data[i] + rhs.data[i]),
        }
    }
}

impl<T: Float, const N: usize> Sub for VecN<T, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            data: std::array::from_fn(|i| self.data[i] - rhs.data[i]),
        }
    }
}

// Elementwise multiplication
impl<T: Float, const N: usize> Mul<VecN<T, N>> for VecN<T, N> {
    type Output = Self;
    fn mul(self, rhs: VecN<T, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| self.data[i] * rhs.data[i]),
        }
    }
}

// Scalar multiplication
// vec * scalar
impl<T: Float, const N: usize> Mul<T> for VecN<T, N> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            data: std::array::from_fn(|i| self.data[i] * rhs),
        }

    }
}

impl<T: Float, const N: usize> Neg for VecN<T, N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut data = self.data;
        for i in 0..N {
            data[i] = -self.data[i];
        }
        Self { data }
    }
}

// Cross product only makes sense for 3D vectors
impl<T: Float> VecN<T, 3> {
    pub fn cross(&self, rhs: &Self) -> Self {
        Self {
            data: [
                self.data[1] * rhs.data[2] - self.data[2] * rhs.data[1],
                self.data[2] * rhs.data[0] - self.data[0] * rhs.data[2],
                self.data[0] * rhs.data[1] - self.data[1] * rhs.data[0],
            ],
        }
    }

    pub fn from_xyz(x: T, y: T, z: T) -> Self {
        Self { data: [x, y, z] }
    }

    pub fn x(&self) -> T {
        self.data[0]
    }

    pub fn y(&self) -> T {
        self.data[1]
    }

    pub fn z(&self) -> T {
        self.data[2]
    }

    pub fn xyz(&self) -> &[T; 3] {
        &self.data
    }
}

// Convenience type aliases
pub type Vec3<T> = VecN<T, 3>;
