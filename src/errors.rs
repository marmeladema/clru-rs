#![allow(missing_docs)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Weight cannot be zero")]
    WeightZero,
    #[error("Weight ({0}) is greater than maximum weight ({1})")]
    WeightTooLarge(usize, usize),
}
