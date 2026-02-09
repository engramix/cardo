/// Chains transformations in a readable left-to-right or right-to-left style.
///
/// Usage:
///   chain!(r1 -> r2 -> r3)  // left-to-right: r1.then(r2).then(r3)
///   chain!(r3 <- r2 <- r1)  // right-to-left: r3.compose(r2).compose(r1)
///
/// Both produce the same result: first r1, then r2, then r3.
#[macro_export]
macro_rules! chain {
    // --- Internal @parse rules ---

    // Found -> : switch to left-to-right mode
    (@parse [$($acc:tt)+] -> $($rest:tt)+) => {
        chain!(@then [$($acc)+] [] $($rest)+)
    };

    // Found <- : switch to right-to-left mode
    (@parse [$($acc:tt)+] <- $($rest:tt)+) => {
        chain!(@compose [$($acc)+] [] $($rest)+)
    };

    // Keep accumulating tokens
    (@parse [$($acc:tt)*] $next:tt $($rest:tt)*) => {
        chain!(@parse [$($acc)* $next] $($rest)*)
    };

    // End of input with no arrow - just return the expression
    (@parse [$($acc:tt)+]) => {
        $($acc)+
    };

    // --- Internal @then rules (left-to-right) ---

    // Found another ->
    (@then [$($left:tt)+] [$($mid:tt)+] -> $($rest:tt)+) => {
        chain!(@then [($($left)+).then($($mid)+)] [] $($rest)+)
    };

    // Accumulate tokens
    (@then [$($left:tt)+] [$($mid:tt)*] $next:tt $($rest:tt)*) => {
        chain!(@then [$($left)+] [$($mid)* $next] $($rest)*)
    };

    // End of input
    (@then [$($left:tt)+] [$($mid:tt)+]) => {
        ($($left)+).then($($mid)+)
    };

    // --- Internal @compose rules (right-to-left) ---

    // Found another <-
    (@compose [$($left:tt)+] [$($mid:tt)+] <- $($rest:tt)+) => {
        chain!(@compose [($($left)+).compose($($mid)+)] [] $($rest)+)
    };

    // Accumulate tokens
    (@compose [$($left:tt)+] [$($mid:tt)*] $next:tt $($rest:tt)*) => {
        chain!(@compose [$($left)+] [$($mid)* $next] $($rest)*)
    };

    // End of input
    (@compose [$($left:tt)+] [$($mid:tt)+]) => {
        ($($left)+).compose($($mid)+)
    };

    // --- Entry point (MUST be last so internal patterns match first) ---
    ($($tokens:tt)+) => {
        chain!(@parse [] $($tokens)+)
    };
}

/// Implements ops for framed vector types.
/// The type must have a `data: [T; N]` field and `_frames: PhantomData<...>` field.
///
/// Usage:
///   impl_framed_vector!(Vector3<F>, 3)          // general + 3D-specific methods
///   impl_framed_vector!(SO3Tangent<A, B, C>, 3) // general + 3D-specific methods
macro_rules! impl_framed_vector {
    ($Type:ident < $($phantom:ident),+ >, $N:tt) => {
        impl_framed_vector!(@general $Type < $($phantom),+ >, $N);
        impl_framed_vector!(@specialize $Type < $($phantom),+ >, $N);
    };

    // --- General ops for any N ---
    (@general $Type:ident < $($phantom:ident),+ >, $N:literal) => {
        impl<$($phantom,)+ T: Float> Add for $Type<$($phantom,)+ T> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self { data: std::array::from_fn(|i| self.data[i] + rhs.data[i]), _frames: PhantomData }
            }
        }

        impl<$($phantom,)+ T: Float> Sub for $Type<$($phantom,)+ T> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self { data: std::array::from_fn(|i| self.data[i] - rhs.data[i]), _frames: PhantomData }
            }
        }

        impl<$($phantom,)+ T: Float> Neg for $Type<$($phantom,)+ T> {
            type Output = Self;
            fn neg(self) -> Self {
                Self { data: std::array::from_fn(|i| -self.data[i]), _frames: PhantomData }
            }
        }

        impl<$($phantom,)+ T: Float> Index<usize> for $Type<$($phantom,)+ T> {
            type Output = T;
            fn index(&self, i: usize) -> &T {
                &self.data[i]
            }
        }

        impl<$($phantom,)+ T: Float> IndexMut<usize> for $Type<$($phantom,)+ T> {
            fn index_mut(&mut self, i: usize) -> &mut T {
                &mut self.data[i]
            }
        }

        // vec * scalar (right multiplication)
        impl<$($phantom,)+ T: Float> Mul<T> for $Type<$($phantom,)+ T> {
            type Output = Self;
            fn mul(self, rhs: T) -> Self {
                Self { data: std::array::from_fn(|i| self.data[i] * rhs), _frames: PhantomData }
            }
        }

        // elementwise multiplication
        impl<$($phantom,)+ T: Float> Mul<$Type<$($phantom,)+ T>> for $Type<$($phantom,)+ T> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                Self { data: std::array::from_fn(|i| self.data[i] * rhs.data[i]), _frames: PhantomData }
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
            pub fn from_data(data: [T; $N]) -> Self {
                Self { data, _frames: PhantomData }
            }

            pub fn zero() -> Self {
                Self {
                    data: [T::zero(); $N],
                    _frames: PhantomData,
                }
            }

            pub fn dot(&self, rhs: &Self) -> T {
                let mut sum = T::zero();
                for i in 0..$N {
                    sum = sum + self.data[i] * rhs.data[i];
                }
                sum
            }

            pub fn norm_squared(&self) -> T {
                self.dot(self)
            }

            pub fn norm(&self) -> T {
                self.norm_squared().sqrt()
            }

            pub fn normalized(&self) -> Self {
                let n = self.norm();
                Self {
                    data: std::array::from_fn(|i| self.data[i] / n),
                    _frames: PhantomData,
                }
            }

            pub fn coeffs(&self) -> &[T] {
                &self.data
            }
        }

        // PartialEq - compare only the data, not the phantom
        impl<$($phantom,)+ T: Float> PartialEq for $Type<$($phantom,)+ T> {
            fn eq(&self, other: &Self) -> bool {
                self.data == other.data
            }
        }

        // Clone
        impl<$($phantom,)+ T: Float> Clone for $Type<$($phantom,)+ T> {
            fn clone(&self) -> Self {
                *self
            }
        }

        // Copy
        impl<$($phantom,)+ T: Float> Copy for $Type<$($phantom,)+ T> {}

        // Debug
        impl<$($phantom,)+ T: Float + std::fmt::Debug> std::fmt::Debug for $Type<$($phantom,)+ T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                // Strip module paths: "crate::module::Type" -> "Type"
                fn short_name(full: &str) -> String {
                    full.split('<')
                        .map(|part| part.rsplit("::").next().unwrap_or(part))
                        .collect::<Vec<_>>()
                        .join("<")
                }
                let name = short_name(std::any::type_name::<Self>());
                // Remove trailing ", f64>" or ", f32>" from type name
                let name = name.trim_end_matches(", f64>").trim_end_matches(", f32>");
                let name = if !name.ends_with('>') { format!("{}>", name) } else { name.to_string() };
                write!(f, "{}{:?}", name, self.coeffs())
            }
        }
    };

    // --- 3D-specific methods ---
    (@specialize $Type:ident < $($phantom:ident),+ >, 3) => {
        impl<$($phantom,)+ T: Float> $Type<$($phantom,)+ T> {
            pub fn new(x: T, y: T, z: T) -> Self {
                Self {
                    data: [x, y, z],
                    _frames: PhantomData,
                }
            }

            pub fn cross(&self, rhs: &Self) -> Self {
                Self {
                    data: [
                        self.data[1] * rhs.data[2] - self.data[2] * rhs.data[1],
                        self.data[2] * rhs.data[0] - self.data[0] * rhs.data[2],
                        self.data[0] * rhs.data[1] - self.data[1] * rhs.data[0],
                    ],
                    _frames: PhantomData,
                }
            }

            pub fn x(&self) -> T {
                self.data[0]
            }

            pub fn y(&self) -> T {
                self.data[1]
            }

            pub fn z(&self) -> T {
                self.data[2]
            }

            pub fn xyz(&self) -> &[T; 3] {
                &self.data
            }
        }
    };

    // Fallback for non-3D sizes
    (@specialize $Type:ident < $($phantom:ident),+ >, $N:literal) => {};
}
