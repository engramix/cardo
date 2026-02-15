use crate::mat::{Mat, Mat3, Mat4, Mat6};
use crate::se3::SE3;
use crate::so3_tangent::SO3Tangent;
use num_traits::Float;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

/// Tangent vector to SE3
///
/// `SE3Tangent<A, B, C>`:
/// - Linear and angular change of frame A relative to frame B, expressed in frame C.
/// - Layout: `[v_x, v_y, v_z, ω_x, ω_y, ω_z]` (linear first, angular second)
#[must_use]
pub struct SE3Tangent<A, B, C, T: Float = f64> {
    pub data: [T; 6],
    _frames: PhantomData<(A, B, C)>,
}

impl_framed_vector!(SE3Tangent<A, B, C>, 6);

impl<A, B, C, T: Float> SE3Tangent<A, B, C, T> {
    pub fn lin(&self) -> [T; 3] {
        [self.data[0], self.data[1], self.data[2]]
    }

    pub fn ang(&self) -> [T; 3] {
        [self.data[3], self.data[4], self.data[5]]
    }

    pub fn from_lin_ang(lin: [T; 3], ang: [T; 3]) -> Self {
        Self::from_data([lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]])
    }

    /// Small adjoint matrix ad(τ) such that [τ₁, τ₂] = ad(τ₁) · τ₂
    /// where [·,·] denotes the Lie bracket.
    ///
    /// ```text
    /// ad(τ) = [[ω]×  [v]×]
    ///         [ 0   [ω]× ]
    /// ```
    pub fn ad(&self) -> Mat6<T> {
        let wx = Mat3::skew(self.ang());
        let vx = Mat3::skew(self.lin());
        let mut m = Mat6::zeros();
        m.set_block(0, 0, &wx);
        m.set_block(0, 3, &vx);
        m.set_block(3, 3, &wx);
        m
    }

    pub fn hat(&self) -> Mat4<T> {
        let mut m = Mat4::zeros();
        m.set_block(0, 0, &Mat3::skew(self.ang()));
        m.set_block(0, 3, &Mat::col(self.lin()));
        m
    }

    pub fn ljac(&self) -> Mat6<T> {
        let so3_ljac = SO3Tangent::<A, B, A, T>::from_data(self.ang()).ljac();
        let q = self.fill_q();
        let mut m = Mat6::zeros();
        m.set_block(0, 0, &so3_ljac);
        m.set_block(0, 3, &q);
        m.set_block(3, 3, &so3_ljac);
        m
    }

    pub fn ljacinv(&self) -> Mat6<T> {
        let so3_ljacinv = SO3Tangent::<A, B, A, T>::from_data(self.ang()).ljacinv();
        let q = self.fill_q();
        let mut m = Mat6::zeros();
        m.set_block(0, 0, &so3_ljacinv);
        m.set_block(0, 3, &(-so3_ljacinv * q * so3_ljacinv));
        m.set_block(3, 3, &so3_ljacinv);
        m
    }

    /// Approximate composition via BCH: log(exp(τ₁) · exp(τ₂)) ≈ τ₁ + τ₂ + ½[τ₁, τ₂]
    /// where [·,·] denotes the Lie bracket.
    pub fn bch_compose(&self, rhs: Self) -> Self {
        let half = T::one() / (T::one() + T::one());
        let bracket = self.ad() * rhs.data;
        Self::from_data(std::array::from_fn(|i| {
            self.data[i] + rhs.data[i] + bracket[i] * half
        }))
    }

    pub fn rjac(&self) -> Mat6<T> {
        let lin = self.lin();
        let ang = self.ang();

        SE3Tangent::<A, B, C, T>::from_lin_ang(
            std::array::from_fn(|i| -lin[i]),
            std::array::from_fn(|i| -ang[i]),
        )
        .ljac()
    }

    pub fn rjacinv(&self) -> Mat6<T> {
        let lin = self.lin();
        let ang = self.ang();

        SE3Tangent::<A, B, C, T>::from_lin_ang(
            std::array::from_fn(|i| -lin[i]),
            std::array::from_fn(|i| -ang[i]),
        )
        .ljacinv()
    }

    fn fill_q(&self) -> Mat3<T> {
        let theta_sq = self.ang().iter().fold(T::zero(), |s, &x| s + x * x);

        let half = T::one() / (T::one() + T::one());
        let a = half;

        let (b, c, d);
        if theta_sq <= T::epsilon() {
            let six = T::from(6.0).unwrap();
            let twenty_four = T::from(24.0).unwrap();
            let sixty = T::from(60.0).unwrap();
            let one_twenty = T::from(120.0).unwrap();
            let seven_twenty = T::from(720.0).unwrap();
            b = T::one() / six + theta_sq / one_twenty;
            c = -T::one() / twenty_four + theta_sq / seven_twenty;
            d = -T::one() / sixty;
        } else {
            let theta = theta_sq.sqrt();
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            let theta_cu = theta_sq * theta;

            b = (theta - sin_t) / theta_cu;
            c = (T::one() - theta_sq * half - cos_t) / (theta_sq * theta_sq);
            let six = T::from(6.0).unwrap();
            let three = T::from(3.0).unwrap();
            d = c - three * (theta - sin_t - theta_cu / six) / (theta_sq * theta_sq * theta);
        }

        let v = Mat3::skew(self.lin());
        let w = Mat3::skew(self.ang());
        let vw = v * w;
        let wv = vw.transpose();
        let wvw = wv * w;
        let vww = vw * w;

        v * a + (wv + vw + wvw) * b
            - (vww - vww.transpose() - wvw * T::from(3.0).unwrap()) * c
            - wvw * w * d
    }
}

impl<A, B, T: Float> SE3Tangent<A, B, A, T> {
    pub fn exp(&self) -> SE3<A, B, T> {
        let so3_tangent = SO3Tangent::<A, B, A, T>::from_data(self.ang());
        SE3::from_vec_quat(so3_tangent.ljac() * self.lin(), so3_tangent.exp().quat)
    }
}
