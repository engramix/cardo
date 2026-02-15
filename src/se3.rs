use crate::act::Act;
use crate::mat::{Mat, Mat3, Mat4, Mat6};
use crate::quat::Quat;
use crate::se3_tangent::SE3Tangent;
use crate::so3::SO3;
use crate::vector3::Vector3;
use num_traits::Float;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Mul;

/// 3D isometry (element of Special Euclidean group).
///
/// `SE3<A, B>`:
/// - Transforms vectors from frame A to frame B
/// - Represents the pose of frame A expressed in frame B
#[must_use]
pub struct SE3<A, B, T: Float = f64> {
    pub vec: [T; 3],
    pub quat: Quat<T>,
    pub(crate) _frames: PhantomData<(A, B)>,
}

impl<A, B, T: Float> PartialEq for SE3<A, B, T> {
    fn eq(&self, other: &Self) -> bool {
        self.quat == other.quat && self.vec == other.vec
    }
}

impl<A, B, T: Float + fmt::Debug> fmt::Debug for SE3<A, B, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (w, x, y, z) = self.quat.wxyz();
        let name_a = std::any::type_name::<A>().rsplit("::").next().unwrap_or("");
        let name_b = std::any::type_name::<B>().rsplit("::").next().unwrap_or("");
        let [tx, ty, tz] = self.vec;
        write!(
            f,
            "SE3<{}, {}>(t=[{:?}, {:?}, {:?}], q=[{:?}, {:?}, {:?}, {:?}])",
            name_a, name_b, tx, ty, tz, w, x, y, z
        )
    }
}

impl<A, B, T: Float> Copy for SE3<A, B, T> {}

impl<A, B, T: Float> Clone for SE3<A, B, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A, B, T: Float> SE3<A, B, T> {
    pub fn from_vec_quat(vec: [T; 3], quat: Quat<T>) -> Self {
        let is_unit = (quat.norm_squared() - T::one()).abs() < T::epsilon().sqrt();
        assert!(is_unit, "quaternion must have unit norm");
        Self {
            quat,
            vec,
            _frames: PhantomData,
        }
    }

    pub fn from_quat(quat: Quat<T>) -> Self {
        Self::from_vec_quat([T::zero(); 3], quat)
    }

    pub fn from_vec(vec: [T; 3]) -> Self {
        Self::from_vec_quat(vec, Quat::identity())
    }

    pub fn identity() -> Self {
        Self::from_vec_quat([T::zero(); 3], Quat::identity())
    }

    pub fn rot_x(angle: T) -> Self {
        let half = angle / (T::one() + T::one());
        Self::from_quat(Quat {
            w: half.cos(),
            x: half.sin(),
            y: T::zero(),
            z: T::zero(),
        })
    }

    pub fn rot_y(angle: T) -> Self {
        let half = angle / (T::one() + T::one());
        Self::from_quat(Quat {
            w: half.cos(),
            x: T::zero(),
            y: half.sin(),
            z: T::zero(),
        })
    }

    pub fn rot_z(angle: T) -> Self {
        let half = angle / (T::one() + T::one());
        Self::from_quat(Quat {
            w: half.cos(),
            x: T::zero(),
            y: T::zero(),
            z: half.sin(),
        })
    }

    pub fn trans_x(d: T) -> Self {
        Self::from_vec([d, T::zero(), T::zero()])
    }

    pub fn trans_y(d: T) -> Self {
        Self::from_vec([T::zero(), d, T::zero()])
    }

    pub fn trans_z(d: T) -> Self {
        Self::from_vec([T::zero(), T::zero(), d])
    }

    pub fn slerp(start: SE3<A, B, T>, end: SE3<A, B, T>, t: T) -> SE3<A, B, T> {
        let delta = start.between(end).log();
        start.rplus(delta * t)
    }

    pub fn inverse(&self) -> SE3<B, A, T> {
        let q_inv = self.quat.conjugate();
        let t_inv = q_inv.rotate(std::array::from_fn(|i| -self.vec[i]));
        SE3::from_vec_quat(t_inv, q_inv)
    }

    pub fn compose<C>(&self, rhs: SE3<C, A, T>) -> SE3<C, B, T> {
        let rot_t = self.quat.rotate(rhs.vec);
        SE3::from_vec_quat(
            std::array::from_fn(|i| self.vec[i] + rot_t[i]),
            self.quat * rhs.quat,
        )
    }

    pub fn then<C>(&self, lhs: SE3<B, C, T>) -> SE3<A, C, T> {
        let rot_t = lhs.quat.rotate(self.vec);
        SE3::from_vec_quat(
            std::array::from_fn(|i| lhs.vec[i] + rot_t[i]),
            lhs.quat * self.quat,
        )
    }

    pub fn log(&self) -> SE3Tangent<A, B, A, T> {
        let so3_tangent = SO3::<A, B, T>::from_quat(self.quat).log();
        SE3Tangent::from_lin_ang(so3_tangent.ljacinv() * self.vec, so3_tangent.data)
    }

    pub fn rplus<C>(&self, rhs: SE3Tangent<C, A, C, T>) -> SE3<C, B, T> {
        self.compose(rhs.exp())
    }

    pub fn rminus<C>(&self, rhs: SE3<C, B, T>) -> SE3Tangent<A, C, A, T> {
        rhs.inverse().compose(*self).log()
    }

    pub fn lplus<C>(&self, lhs: SE3Tangent<B, C, B, T>) -> SE3<A, C, T> {
        self.then(lhs.exp())
    }

    pub fn lminus<C>(&self, rhs: SE3<A, C, T>) -> SE3Tangent<C, B, C, T> {
        self.compose(rhs.inverse()).log()
    }

    /// Relative rotation and translation of `other` wrt `self`
    pub fn between<C>(&self, other: SE3<C, B, T>) -> SE3<C, A, T> {
        self.inverse().compose(other)
    }

    pub fn adjoint<C>(&self, v: SE3Tangent<C, A, C, T>) -> SE3Tangent<A, B, A, T> {
        // NOTE: exp(tau<B, E, B>) * X_lhs<A, B> = X_rhs<B, E> * exp(tau<A, B, A>)
        SE3Tangent::from_data(self.adjoint_matrix() * v.data)
    }

    pub fn adjoint_matrix(&self) -> Mat6<T> {
        let r = SO3::<A, B, T>::from_quat(self.quat).to_matrix();
        let tx = Mat3::skew(self.vec);
        let mut ad = Mat6::zeros();
        ad.set_block(0, 0, &r);
        ad.set_block(0, 3, &(tx * r));
        ad.set_block(3, 3, &r);
        ad
    }

    pub fn to_matrix(&self) -> Mat4<T> {
        let r = SO3::<A, B, T>::from_quat(self.quat).to_matrix();
        let mut m = Mat4::zeros();
        m.set_block(0, 0, &r);
        m.set_block(0, 3, &Mat::col(self.vec));
        m.data[3][3] = T::one();
        m
    }
}

// Compose: SE3<A,B> * SE3<C,A> -> SE3<C,B>
impl<A, B, C, T: Float> Mul<SE3<C, A, T>> for SE3<A, B, T> {
    type Output = SE3<C, B, T>;
    fn mul(self, rhs: SE3<C, A, T>) -> SE3<C, B, T> {
        self.compose(rhs)
    }
}

// Act on Vector3: SE3<A,B> * Vector3<A> -> Vector3<B>
impl<A, B, T: Float> Act<Vector3<A, T>> for SE3<A, B, T> {
    type Output = Vector3<B, T>;
    fn act(&self, v: Vector3<A, T>) -> Vector3<B, T> {
        let r = self.quat.rotate(v.data);
        Vector3::from_data(std::array::from_fn(|i| r[i] + self.vec[i]))
    }
}

// Act on SE3Tangent: SE3<A,B> * SE3Tangent<X,Y,A> -> SE3Tangent<X,Y,B>
impl<A, B, X, Y, T: Float> Act<SE3Tangent<X, Y, A, T>> for SE3<A, B, T> {
    type Output = SE3Tangent<X, Y, B, T>;
    fn act(&self, v: SE3Tangent<X, Y, A, T>) -> SE3Tangent<X, Y, B, T> {
        SE3Tangent::from_lin_ang(self.quat.rotate(v.lin()), self.quat.rotate(v.ang()))
    }
}

impl<A, B, T: Float> Mul<Vector3<A, T>> for SE3<A, B, T> {
    type Output = Vector3<B, T>;
    fn mul(self, rhs: Vector3<A, T>) -> Vector3<B, T> {
        self.act(rhs)
    }
}

impl<A, B, X, Y, T: Float> Mul<SE3Tangent<X, Y, A, T>> for SE3<A, B, T> {
    type Output = SE3Tangent<X, Y, B, T>;
    fn mul(self, rhs: SE3Tangent<X, Y, A, T>) -> SE3Tangent<X, Y, B, T> {
        self.act(rhs)
    }
}
