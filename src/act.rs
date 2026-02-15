/// Group action — re-express a value in a new frame.
pub trait Act<V> {
    type Output;
    fn act(&self, v: V) -> Self::Output;
}

/// Group action — re-express a value in a new frame and calculate jacobians.
pub trait ActWithJac<V>: Act<V> {
    type JacGroup;
    type JacInput;
    fn act_with_jac(&self, v: V) -> (<Self as Act<V>>::Output, Self::JacGroup, Self::JacInput);
}
