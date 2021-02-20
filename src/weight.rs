/// Trait used to retrieve the weight of a value.
pub trait WeightScale<V> {
    /// Returns the weight of a value.
    fn weight(&self, value: &V) -> usize;
}

/// A scale that always return 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct ZeroWeightScale;

impl<V> WeightScale<V> for ZeroWeightScale {
    #[inline]
    fn weight(&self, _: &V) -> usize {
        0
    }
}
