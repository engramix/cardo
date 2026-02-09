use crate::framed::Vector3;
use crate::primitives::{Quat, Vec3};
use num_traits::Float;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

/// 3D rotation (element of Special Orthogonal group).
///
/// `SO3<A, B>`:
/// - Transforms vectors from frame A to frame B
/// - Represents the orientation of frame A expressed in frame B
#[derive(PartialEq)]
pub struct SO3<A, B, T: Float = f64> {
    pub quat: Quat<T>,
    pub(crate) _frames: PhantomData<(A, B)>,
}

impl<A, B, T: Float + fmt::Debug> fmt::Debug for SO3<A, B, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (w, x, y, z) = self.quat.wxyz();
        write!(
            f,
            "SO3<{}, {}>[{:?}, {:?}, {:?}, {:?}]",
            std::any::type_name::<A>(),
            std::any::type_name::<B>(),
            w,
            x,
            y,
            z
        )
    }
}

impl<A, B, T: Float> Copy for SO3<A, B, T> {}

impl<A, B, T: Float> Clone for SO3<A, B, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Tangent vector to SO3
///
/// `SO3Tangent<A, B, C>`:
/// - Angular change (e.g. angular velocity) of frame A relative to frame B, expressed in frame C.
#[derive(PartialEq)]
pub struct SO3Tangent<A, B, C, T: Float = f64> {
    pub vec: Vec3<T>,
    pub(crate) _frames: PhantomData<(A, B, C)>,
}

impl<A, B, C, T: Float + fmt::Debug> fmt::Debug for SO3Tangent<A, B, C, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SO3Tangent<{}, {}, {}>{:?}",
            std::any::type_name::<A>(),
            std::any::type_name::<B>(),
            std::any::type_name::<C>(),
            self.vec.data
        )
    }
}

impl<A, B, C, T: Float> Copy for SO3Tangent<A, B, C, T> {}

impl<A, B, C, T: Float> Clone for SO3Tangent<A, B, C, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl_framed_vector_ops!(SO3Tangent<A, B, C>, Vec3);

impl<A, B, T: Float> SO3<A, B, T> {
    pub fn from_quat(quat: Quat<T>) -> Self {
        assert!(
            (quat.norm_squared() - T::one()).abs() < T::epsilon().sqrt(),
            "quaternion must have unit norm"
        );
        Self {
            quat,
            _frames: PhantomData,
        }
    }

    pub fn from_axis_angle(axis: &Vector3<A, T>, angle: T) -> Self {
        assert!(
            (axis.norm_squared() - T::one()).abs() < T::epsilon().sqrt(),
            "axis must have unit norm"
        );
        let half = angle / (T::one() + T::one());
        let s = half.sin();
        Self::from_quat(Quat {
            w: half.cos(),
            x: axis.vec.x() * s,
            y: axis.vec.y() * s,
            z: axis.vec.z() * s,
        })
    }

    pub fn identity() -> Self {
        Self::from_quat(Quat::identity())
    }

    pub fn exp(v: &SO3Tangent<A, B, A, T>) -> Self {
        let angle = v.dot(v).sqrt();

        if angle < T::epsilon() {
            let half = T::one() / (T::one() + T::one());
            Self::from_quat(Quat {
                w: T::one(),
                x: v.x() * half,
                y: v.y() * half,
                z: v.z() * half,
            })
        } else {
            let half_angle = angle / (T::one() + T::one());
            let k = half_angle.sin() / angle;
            Self::from_quat(Quat {
                w: half_angle.cos(),
                x: v.x() * k,
                y: v.y() * k,
                z: v.z() * k,
            })
        }
    }

    pub fn log(&self) -> SO3Tangent<A, B, A, T> {
        let (w, x, y, z) = self.quat.wxyz();
        let sin_angle_sq = x * x + y * y + z * z;

        let log_coeff = if sin_angle_sq < T::epsilon() {
            T::one()
        } else {
            let sin_angle = sin_angle_sq.sqrt();
            let cos_angle = w;
            let angle = if cos_angle.is_sign_negative() {
                sin_angle.neg().atan2(cos_angle.neg())
            } else {
                sin_angle.atan2(cos_angle)
            };
            angle / sin_angle
        };

        SO3Tangent {
            vec: Vec3::from_xyz(x, y, z) * (T::one() + T::one()) * log_coeff,
            _frames: PhantomData,
        }
    }

    pub fn inverse(&self) -> SO3<B, A, T> {
        SO3::from_quat(self.quat.conjugate())
    }

    pub fn compose<C>(self, rhs: SO3<C, A, T>) -> SO3<C, B, T> {
        SO3::from_quat(self.quat * rhs.quat)
    }

    pub fn then<C>(self, lhs: SO3<B, C, T>) -> SO3<A, C, T> {
        SO3::from_quat(lhs.quat * self.quat)
    }

    pub fn act(&self, v: Vector3<A, T>) -> Vector3<B, T> {
        Vector3 {
            vec: self.quat.rotate(&v.vec),
            _frames: PhantomData,
        }
    }

    pub fn to_matrix(&self) -> [[T; 3]; 3] {
        let Quat { w, x, y, z } = self.quat;
        let two = T::one() + T::one();
        [
            [
                T::one() - two * (y * y + z * z),
                two * (x * y - w * z),
                two * (x * z + w * y),
            ],
            [
                two * (x * y + w * z),
                T::one() - two * (x * x + z * z),
                two * (y * z - w * x),
            ],
            [
                two * (x * z - w * y),
                two * (y * z + w * x),
                T::one() - two * (x * x + y * y),
            ],
        ]
    }
}

// Compose: SO3<A,B> * SO3<C,A> -> SO3<C,B>
impl<A, B, C, T: Float> Mul<SO3<C, A, T>> for SO3<A, B, T> {
    type Output = SO3<C, B, T>;
    fn mul(self, rhs: SO3<C, A, T>) -> SO3<C, B, T> {
        self.compose(rhs)
    }
}

// Act: SO3<A,B> * Vector3<A> -> Vector3<B>
impl<A, B, T: Float> Mul<Vector3<A, T>> for SO3<A, B, T> {
    type Output = Vector3<B, T>;
    fn mul(self, rhs: Vector3<A, T>) -> Vector3<B, T> {
        self.act(rhs)
    }
}
