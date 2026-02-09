/// Implements ops for framed vector types.
/// The type must have a `vec` field and `_frames: PhantomData<...>` field.
///
/// Usage:
///   impl_framed_vector_ops!(Vector3<F>, Vec3);      // general + 3D-specific methods
///   impl_framed_vector_ops!(SE3Tangent<A, B, C>);   // general ops only
macro_rules! impl_framed_vector_ops {
    // Vec3: general ops + 3D-specific methods
    ($Type:ident < $($phantom:ident),+ >, Vec3) => {
        impl_framed_vector_ops!($Type < $($phantom),+ >);

        // 3D-specific methods
        impl<$($phantom,)+ T: Float> $Type<$($phantom,)+ T> {
            pub fn new(x: T, y: T, z: T) -> Self {
                Self {
                    vec: Vec3::from_xyz(x, y, z),
                    _frames: PhantomData,
                }
            }

            pub fn cross(&self, rhs: &Self) -> Self {
                Self {
                    vec: self.vec.cross(&rhs.vec),
                    _frames: PhantomData,
                }
            }

            pub fn x(&self) -> T {
                self.vec.x()
            }

            pub fn y(&self) -> T {
                self.vec.y()
            }

            pub fn z(&self) -> T {
                self.vec.z()
            }

            pub fn xyz(&self) -> &[T; 3] {
                &self.vec.data
            }
        }
    };

    // General: arithmetic ops + vector methods
    ($Type:ident < $($phantom:ident),+ >) => {
        impl<$($phantom,)+ T: Float> Add for $Type<$($phantom,)+ T> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self { vec: self.vec + rhs.vec, _frames: PhantomData }
            }
        }

        impl<$($phantom,)+ T: Float> Sub for $Type<$($phantom,)+ T> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self { vec: self.vec - rhs.vec, _frames: PhantomData }
            }
        }

        impl<$($phantom,)+ T: Float> Mul<T> for $Type<$($phantom,)+ T> {
            type Output = Self;
            fn mul(self, rhs: T) -> Self {
                Self { vec: self.vec * rhs, _frames: PhantomData }
            }
        }

        impl<$($phantom,)+ T: Float> Neg for $Type<$($phantom,)+ T> {
            type Output = Self;
            fn neg(self) -> Self {
                Self { vec: -self.vec, _frames: PhantomData }
            }
        }

        // scalar * vec (left multiplication)
        impl<$($phantom),+> Mul<$Type<$($phantom,)+ f64>> for f64 {
            type Output = $Type<$($phantom,)+ f64>;
            fn mul(self, rhs: $Type<$($phantom,)+ f64>) -> $Type<$($phantom,)+ f64> {
                rhs * self
            }
        }

        impl<$($phantom),+> Mul<$Type<$($phantom,)+ f32>> for f32 {
            type Output = $Type<$($phantom,)+ f32>;
            fn mul(self, rhs: $Type<$($phantom,)+ f32>) -> $Type<$($phantom,)+ f32> {
                rhs * self
            }
        }

        // General vector methods
        impl<$($phantom,)+ T: Float> $Type<$($phantom,)+ T> {
            pub fn zero() -> Self {
                Self {
                    vec: crate::primitives::VecN::zero(),
                    _frames: PhantomData,
                }
            }

            pub fn dot(&self, rhs: &Self) -> T {
                self.vec.dot(&rhs.vec)
            }

            pub fn norm_squared(&self) -> T {
                self.vec.norm_squared()
            }

            pub fn norm(&self) -> T {
                self.vec.norm()
            }

            pub fn normalized(&self) -> Self {
                Self {
                    vec: self.vec.normalized(),
                    _frames: PhantomData,
                }
            }

            pub fn coeffs(&self) -> &[T] {
                &self.vec.data
            }
        }
    };
}
