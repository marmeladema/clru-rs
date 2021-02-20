/// Trait used to retrieve the weight of a value.
pub trait WeightScale<K, V> {
    /// Returns the weight of a value.
    fn weight(&self, key: &K, value: &V) -> usize;
}

/// A scale that always return 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct ZeroWeightScale;

impl<K, V> WeightScale<K, V> for ZeroWeightScale {
    #[inline]
    fn weight(&self, _: &K, _: &V) -> usize {
        0
    }
}
