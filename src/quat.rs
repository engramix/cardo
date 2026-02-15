use num_traits::Float;
use std::ops::Mul;

#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat<T = f64> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Float> Quat<T> {
    pub fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }

    pub fn wxyz(&self) -> (T, T, T, T) {
        (self.w, self.x, self.y, self.z)
    }

    pub fn identity() -> Self {
        Self {
            w: T::one(),
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn norm_squared(&self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        Self {
            w: self.w / n,
            x: self.x / n,
            y: self.y / n,
            z: self.z / n,
        }
    }

    pub(crate) fn rotate(&self, v: [T; 3]) -> [T; 3] {
        let p = Quat {
            w: T::zero(),
            x: v[0],
            y: v[1],
            z: v[2],
        };
        let res = *self * p * self.conjugate();
        [res.x, res.y, res.z]
    }
}

impl<T: Float> Mul for Quat<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (pw, px, py, pz) = self.wxyz();
        let (qw, qx, qy, qz) = rhs.wxyz();
        Self {
            w: pw * qw - px * qx - py * qy - pz * qz,
            x: pw * qx + px * qw + py * qz - pz * qy,
            y: pw * qy - px * qz + py * qw + pz * qx,
            z: pw * qz + px * qy - py * qx + pz * qw,
        }
    }
}
