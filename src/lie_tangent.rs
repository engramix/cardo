/// Implements derived Lie tangent methods that are identical across all tangent types.
///
/// Usage:
///   impl_lie_tangent!(Tangent = SO3Tangent, AdjMat = Mat3);
///   impl_lie_tangent!(Tangent = SE3Tangent, AdjMat = Mat6);
macro_rules! impl_lie_tangent {
    (Tangent = $Tangent:ident, AdjMat = $AdjMat:ident) => {
        impl<A, B, C, T: Float> $Tangent<A, B, C, T> {
            pub fn rjac(&self) -> $AdjMat<T> {
                (-*self).ljac()
            }

            pub fn rjacinv(&self) -> $AdjMat<T> {
                (-*self).ljacinv()
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
        }
    };
}
