/// Implements derived Lie group methods that are identical across all groups.
///
/// Usage:
///   impl_lie_group!(Group = SO3, Tangent = SO3Tangent, AdjMat = Mat3);
///   impl_lie_group!(Group = SE3, Tangent = SE3Tangent, AdjMat = Mat6);
macro_rules! impl_lie_group {
    (Group = $Group:ident, Tangent = $Tangent:ident, AdjMat = $AdjMat:ident) => {
        impl<A, B, T: Float> $Group<A, B, T> {
            pub fn then<C>(&self, lhs: $Group<B, C, T>) -> $Group<A, C, T> {
                lhs.compose(*self)
            }

            /// Relative transformation of `other` with respect to `self`.
            pub fn between<C>(&self, other: $Group<C, B, T>) -> $Group<C, A, T> {
                self.inverse().compose(other)
            }

            pub fn slerp(start: $Group<A, B, T>, end: $Group<A, B, T>, t: T) -> $Group<A, B, T> {
                let delta = start.between(end).log();
                start.rplus(delta * t)
            }

            pub fn rplus<C>(&self, rhs: $Tangent<C, A, C, T>) -> $Group<C, B, T> {
                self.compose(rhs.exp())
            }

            pub fn rminus<C>(&self, rhs: $Group<C, B, T>) -> $Tangent<A, C, A, T> {
                rhs.inverse().compose(*self).log()
            }

            pub fn lplus<C>(&self, lhs: $Tangent<B, C, B, T>) -> $Group<A, C, T> {
                self.then(lhs.exp())
            }

            pub fn lminus<C>(&self, rhs: $Group<A, C, T>) -> $Tangent<C, B, C, T> {
                self.compose(rhs.inverse()).log()
            }

            // exp(tau<B, E, B>) * X_lhs<A, B> = X_rhs<B, E> * exp(tau<A, B, A>)
            pub fn adjoint<C>(&self, v: $Tangent<C, A, C, T>) -> $Tangent<A, B, A, T> {
                $Tangent::from_data(self.adjoint_matrix() * v.data)
            }

            pub fn log_with_jac(&self) -> ($Tangent<A, B, A, T>, $AdjMat<T>) {
                let result = self.log();
                (result, result.rjacinv())
            }

            pub fn inverse_with_jac(&self) -> ($Group<B, A, T>, $AdjMat<T>) {
                (self.inverse(), -self.adjoint_matrix())
            }

            pub fn compose_with_jac<C>(
                &self,
                rhs: $Group<C, A, T>,
            ) -> ($Group<C, B, T>, $AdjMat<T>, $AdjMat<T>) {
                (
                    self.compose(rhs),
                    rhs.inverse().adjoint_matrix(),
                    $AdjMat::identity(),
                )
            }

            pub fn rplus_with_jac<C>(
                &self,
                rhs: $Tangent<C, A, C, T>,
            ) -> ($Group<C, B, T>, $AdjMat<T>, $AdjMat<T>) {
                (
                    self.rplus(rhs),
                    rhs.exp().inverse().adjoint_matrix(),
                    rhs.rjac(),
                )
            }

            pub fn rminus_with_jac<C>(
                &self,
                rhs: $Group<C, B, T>,
            ) -> ($Tangent<A, C, A, T>, $AdjMat<T>, $AdjMat<T>) {
                let result = self.rminus(rhs);
                (result, result.rjacinv(), -result.ljacinv())
            }

            pub fn lplus_with_jac<C>(
                &self,
                lhs: $Tangent<B, C, B, T>,
            ) -> ($Group<A, C, T>, $AdjMat<T>, $AdjMat<T>) {
                (
                    self.lplus(lhs),
                    $AdjMat::identity(),
                    self.inverse().adjoint_matrix() * lhs.rjac(),
                )
            }

            pub fn lminus_with_jac<C>(
                &self,
                rhs: $Group<A, C, T>,
            ) -> ($Tangent<C, B, C, T>, $AdjMat<T>, $AdjMat<T>) {
                let result = self.lminus(rhs);
                let rhs_adj = rhs.adjoint_matrix();
                (
                    result,
                    result.rjacinv() * rhs_adj,
                    -result.rjacinv() * rhs_adj,
                )
            }
        }

        // Compose: Group<A,B> * Group<C,A> -> Group<C,B>
        impl<A, B, C, T: Float> Mul<$Group<C, A, T>> for $Group<A, B, T> {
            type Output = $Group<C, B, T>;
            fn mul(self, rhs: $Group<C, A, T>) -> $Group<C, B, T> {
                self.compose(rhs)
            }
        }

        // Act on Vector3: Group<A,B> * Vector3<A> -> Vector3<B>
        impl<A, B, T: Float> Mul<Vector3<A, T>> for $Group<A, B, T> {
            type Output = Vector3<B, T>;
            fn mul(self, rhs: Vector3<A, T>) -> Vector3<B, T> {
                self.act(rhs)
            }
        }

        // Act on Tangent: Group<A,B> * Tangent<X,Y,A> -> Tangent<X,Y,B>
        impl<A, B, X, Y, T: Float> Mul<$Tangent<X, Y, A, T>> for $Group<A, B, T> {
            type Output = $Tangent<X, Y, B, T>;
            fn mul(self, rhs: $Tangent<X, Y, A, T>) -> $Tangent<X, Y, B, T> {
                self.act(rhs)
            }
        }
    };
}
