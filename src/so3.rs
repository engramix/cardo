use crate::mat::Mat3;
use crate::quat::Quat;
use crate::vector3::Vector3;
use num_traits::Float;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

/// 3D rotation (element of Special Orthogonal group).
///
/// `SO3<A, B>`:
/// - Transforms vectors from frame A to frame B
/// - Represents the orientation of frame A expressed in frame B
#[must_use]
#[derive(PartialEq)]
pub struct SO3<A, B, T: Float = f64> {
    pub quat: Quat<T>,
    pub(crate) _frames: PhantomData<(A, B)>,
}

impl<A, B, T: Float + fmt::Debug> fmt::Debug for SO3<A, B, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (w, x, y, z) = self.quat.wxyz();
        let name_a = std::any::type_name::<A>().rsplit("::").next().unwrap_or("");
        let name_b = std::any::type_name::<B>().rsplit("::").next().unwrap_or("");
        write!(
            f,
            "SO3<{}, {}>(w={:?}, x={:?}, y={:?}, z={:?})",
            name_a, name_b, w, x, y, z
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
#[must_use]
pub struct SO3Tangent<A, B, C, T: Float = f64> {
    pub data: [T; 3],
    _frames: PhantomData<(A, B, C)>,
}

impl_framed_vector!(SO3Tangent<A, B, C>, 3);

impl<A, B, T: Float> SO3<A, B, T> {
    pub fn from_quat(quat: Quat<T>) -> Self {
        let is_unit = (quat.norm_squared() - T::one()).abs() < T::epsilon().sqrt();
        assert!(is_unit, "quaternion must have unit norm");
        Self {
            quat,
            _frames: PhantomData,
        }
    }

    pub fn from_axis_angle(axis: &Vector3<A, T>, angle: T) -> Self {
        let is_unit = (axis.norm_squared() - T::one()).abs() < T::epsilon().sqrt();
        assert!(is_unit, "axis must have unit norm");
        let half = angle / (T::one() + T::one());
        let s = half.sin();
        Self::from_quat(Quat {
            w: half.cos(),
            x: axis.x() * s,
            y: axis.y() * s,
            z: axis.z() * s,
        })
    }

    pub fn identity() -> Self {
        Self::from_quat(Quat::identity())
    }

    pub fn rot_x(angle: T) -> Self {
        Self::from_axis_angle(&Vector3::unit_x(), angle)
    }

    pub fn rot_y(angle: T) -> Self {
        Self::from_axis_angle(&Vector3::unit_y(), angle)
    }

    pub fn rot_z(angle: T) -> Self {
        Self::from_axis_angle(&Vector3::unit_z(), angle)
    }

    /// Spherical linear interpolation between two rotations.
    ///
    /// ```
    /// # use cardo::prelude::*;
    /// struct Cam;
    /// struct World;
    ///
    /// let start: SO3<Cam, World> = SO3::identity();
    /// let end: SO3<Cam, World> = SO3::rot_z(1.0);
    ///
    /// let mid = SO3::slerp(start, end, 0.5);
    /// ```
    pub fn slerp(start: SO3<A, B, T>, end: SO3<A, B, T>, t: T) -> SO3<A, B, T> {
        let delta = start.between(end).log();
        start.rplus(delta * t)
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

        let scale = if sin_angle_sq < T::epsilon() {
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

        let two = T::one() + T::one();
        SO3Tangent::new(x * two * scale, y * two * scale, z * two * scale)
    }

    pub fn inverse(&self) -> SO3<B, A, T> {
        SO3::from_quat(self.quat.conjugate())
    }

    pub fn compose<C>(&self, rhs: SO3<C, A, T>) -> SO3<C, B, T> {
        SO3::from_quat(self.quat * rhs.quat)
    }

    pub fn then<C>(&self, lhs: SO3<B, C, T>) -> SO3<A, C, T> {
        SO3::from_quat(lhs.quat * self.quat)
    }

    pub fn act(&self, v: Vector3<A, T>) -> Vector3<B, T> {
        Vector3::from_data(self.quat.rotate(&v.data))
    }

    /// Perturb a rotation in the local frame.
    ///
    /// # Examples
    ///
    /// Basic gyro integration
    ///
    /// ```
    /// # use cardo::prelude::*;
    /// struct Body;
    /// struct World;
    ///
    /// let orientation: SO3<Body, World> = SO3::identity();
    ///
    /// let angvel: SO3Tangent<Body, Body, Body> = SO3Tangent::new(0.01, 0.1, 0.1);
    /// let dt = 0.01;
    ///
    /// let updated = orientation.rplus(angvel * dt);
    /// ```
    pub fn rplus<C>(&self, rhs: SO3Tangent<C, A, C, T>) -> SO3<C, B, T> {
        self.compose(SO3::exp(&rhs))
    }

    pub fn rminus<C>(&self, rhs: SO3<C, B, T>) -> SO3Tangent<A, C, A, T> {
        rhs.inverse().compose(*self).log()
    }

    pub fn lplus<C>(&self, lhs: SO3Tangent<B, C, B, T>) -> SO3<A, C, T> {
        self.then(SO3::exp(&lhs))
    }

    pub fn lminus<C>(&self, rhs: SO3<A, C, T>) -> SO3Tangent<C, B, C, T> {
        self.compose(rhs.inverse()).log()
    }

    /// Relative rotation of `other` wrt `self`
    pub fn between<C>(&self, other: SO3<C, B, T>) -> SO3<C, A, T> {
        self.inverse().compose(other)
    }

    pub fn adjoint<C>(&self, v: SO3Tangent<C, A, C, T>) -> SO3Tangent<A, B, A, T> {
        // NOTE: exp(tau<B, E, B>) * X_lhs<A, B> = X_rhs<B, E> * exp(tau<A, B, A>)
        SO3Tangent::from_data(self.adjoint_matrix() * v.data)
    }

    pub fn adjoint_matrix(&self) -> Mat3<T> {
        self.to_matrix()
    }

    pub fn to_matrix(&self) -> Mat3<T> {
        let Quat { w, x, y, z } = self.quat;
        let two = T::one() + T::one();
        Mat3::from_data([
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
        ])
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

// Re-express tangent: SO3<A,B> * SO3Tangent<X,Y,A> -> SO3Tangent<X,Y,B>
impl<A, B, X, Y, T: Float> Mul<SO3Tangent<X, Y, A, T>> for SO3<A, B, T> {
    type Output = SO3Tangent<X, Y, B, T>;
    fn mul(self, rhs: SO3Tangent<X, Y, A, T>) -> SO3Tangent<X, Y, B, T> {
        SO3Tangent::from_data(self.quat.rotate(&rhs.data))
    }
}
