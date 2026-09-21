//! Traits for calculating the two kinds of quantities.

pub mod classical;
pub mod quantum;

/// An enum representing the two possible estimator types - those which yield values (e.g., the energy estimator)
/// and those which yield vectors (e.g., the angular momentum estimator).
pub enum Estimator<T, U> {
    /// A value estimator.
    Value(T),
    /// A vector estimator.
    Vector(U),
}
