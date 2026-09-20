//! Traits for calculating the two kinds of quantities.

pub mod classical;
pub mod quantum;

pub enum Estimator<T, U> {
    Value(T),
    Vector(U),
}
