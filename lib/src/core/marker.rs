//! Marker traits for allowing default implementations.

/// A marker trait used to exclude `()`.
pub trait MeaningfulOutput {}

impl !MeaningfulOutput for () {}

/// A trait for which `T: ValidOutput<T>` and `(): ValidOutput<T>` for every type `T`.
pub trait ValidOutput<T> {}

impl<T: MeaningfulOutput> ValidOutput<T> for T {}

impl<T> ValidOutput<T> for () {}
