use num_traits::Float;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

#[must_use]
pub struct Vector3<F, T: Float = f64> {
    pub data: [T; 3],
    _frames: PhantomData<F>,
}

impl_framed_vector!(Vector3<F>, 3);
