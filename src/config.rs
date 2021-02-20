use crate::weight::{WeightScale, ZeroWeightScale};
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

/// A configuration structure used to create an LRU cache.
pub struct CLruCacheConfig<V, S = RandomState, W = ZeroWeightScale> {
    pub(crate) capacity: NonZeroUsize,
    pub(crate) hash_builder: S,
    pub(crate) reserve: Option<usize>,
    pub(crate) scale: W,
    _marker: PhantomData<V>,
}

impl<V> CLruCacheConfig<V> {
    /// Creates a new configuration that will create an LRU cache
    /// that will hold at most `capacity` elements and default parameters.
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            hash_builder: RandomState::default(),
            reserve: None,
            scale: ZeroWeightScale,
            _marker: PhantomData,
        }
    }
}

impl<V, S: BuildHasher, W: WeightScale<V>> CLruCacheConfig<V, S, W> {
    /// Configure the provided hash builder.
    pub fn with_hasher<O: BuildHasher>(self, hash_builder: O) -> CLruCacheConfig<V, O, W> {
        let Self {
            capacity,
            reserve,
            scale,
            _marker,
            ..
        } = self;
        CLruCacheConfig {
            capacity,
            hash_builder,
            reserve,
            scale,
            _marker,
        }
    }

    /// Configure the amount of pre-allocated memory to order to hold at least `reserve` elements
    /// without reallocating.
    pub fn with_memory(mut self, reserve: usize) -> Self {
        self.reserve = Some(reserve);
        self
    }

    /// Configure the provided scale.
    pub fn with_scale<O: WeightScale<V>>(self, scale: O) -> CLruCacheConfig<V, S, O> {
        let Self {
            capacity,
            hash_builder,
            reserve,
            ..
        } = self;
        CLruCacheConfig {
            capacity,
            hash_builder,
            reserve,
            scale,
            _marker: PhantomData,
        }
    }
}
