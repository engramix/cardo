use num_traits::Float;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use crate::primitives::Vec3;

pub struct Vector3<F, T: Float = f64> {
    pub(crate) vec: Vec3<T>,
    pub(crate) _frames: PhantomData<F>,
}

impl_framed_vector_ops!(Vector3<F>, Vec3);
