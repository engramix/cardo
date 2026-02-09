use num_traits::Float;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use crate::primitives::Vec3;

#[derive(PartialEq)]
pub struct Vector3<F, T: Float = f64> {
    pub(crate) vec: Vec3<T>,
    pub(crate) _frames: PhantomData<F>,
}

impl<F, T: Float + fmt::Debug> fmt::Debug for Vector3<F, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Vector3<{}>{:?}",
            std::any::type_name::<F>(),
            self.vec.data
        )
    }
}

impl<F, T: Float> Copy for Vector3<F, T> {}

impl<F, T: Float> Clone for Vector3<F, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl_framed_vector_ops!(Vector3<F>, Vec3);
