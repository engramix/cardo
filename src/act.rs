/// Group action — re-express a value in a new frame.
pub trait Act<V> {
    type Output;
    fn act(&self, v: V) -> Self::Output;
}
