use crate::mat::Mat3;
use crate::quat::Quat;
use crate::so3::SO3;
use num_traits::Float;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

/// Tangent vector to SO3
///
/// `SO3Tangent<A, B, C>`:
/// - Angular change (e.g. angular velocity) of frame A relative to frame B, expressed in frame C.
#[must_use]
pub struct SO3Tangent<A, B, C, T: Float = f64> {
    pub data: [T; 3],
    _frames: PhantomData<(A, B, C)>,
}

impl_framed_vector!(SO3Tangent<A, B, C>, 3);

impl<A, B, C, T: Float> SO3Tangent<A, B, C, T> {
    pub fn hat(&self) -> Mat3<T> {
        Mat3::skew(self.data)
    }

    /// Small adjoint matrix ad(ω) such that [ω₁, ω₂] = ad(ω₁) · ω₂
    /// where [·,·] denotes the Lie bracket.
    ///
    /// For SO3, ad(ω) = [ω]× = hat(ω)
    pub fn ad(&self) -> Mat3<T> {
        self.hat()
    }

    pub fn ljac(&self) -> Mat3<T> {
        let theta_sq = self.norm_squared();

        let w = self.hat();

        if theta_sq < T::epsilon() {
            let half = T::one() / (T::one() + T::one());
            return Mat3::identity() + w * half;
        }

        let theta = theta_sq.sqrt();

        Mat3::identity()
            + w * ((T::one() - theta.cos()) / theta_sq)
            + w * w * ((theta - theta.sin()) / (theta_sq * theta))
    }

    pub fn ljacinv(&self) -> Mat3<T> {
        let theta_sq = self.norm_squared();
        let w = self.hat();
        let half = T::one() / (T::one() + T::one());

        if theta_sq < T::epsilon() {
            return Mat3::identity() - w * half;
        }

        let theta = theta_sq.sqrt();
        let two = T::one() + T::one();
        let k = (T::one() + theta.cos()) / (two * theta.sin());

        Mat3::identity() - w * half + w * w * (T::one() / theta_sq - k / theta)
    }

}

impl_lie_tangent!(Tangent = SO3Tangent, AdjMat = Mat3);

impl<A, B, T: Float> SO3Tangent<A, B, A, T> {
    pub fn exp(&self) -> SO3<A, B, T> {
        let angle = self.dot(self).sqrt();

        if angle < T::epsilon() {
            let half = T::one() / (T::one() + T::one());
            SO3::from_quat(Quat {
                w: T::one(),
                x: self.x() * half,
                y: self.y() * half,
                z: self.z() * half,
            })
        } else {
            let half_angle = angle / (T::one() + T::one());
            let k = half_angle.sin() / angle;
            SO3::from_quat(Quat {
                w: half_angle.cos(),
                x: self.x() * k,
                y: self.y() * k,
                z: self.z() * k,
            })
        }
    }
}
