//! Traits for calculating quantum observables.

use crate::core::{GroupInTypeInImage, marker::ValidOutput};
use macros::heavy_computation;
use std::sync::{Barrier, RwLock};

mod atom_additive;
pub use atom_additive::{
    AdditiveValueQuantumEstimator, AdditiveVectorQuantumEstimator, AtomAdditiveQuantumEstimator,
};

mod atom_multiplicative;
pub use atom_multiplicative::{
    AtomMultiplicativeQuantumEstimator, MultiplicativeValueQuantumEstimator,
};

/// A trait for quantum estimators.
///
/// All contributions returned by estimators in all images are averaged-out to produce
/// the final value.
///
/// The generic parameter `O` is the type of the values returned by calculations.
/// Setting it to `()` implies that the calculations are sent to another estimator
/// that combines the recieved data and returns the final observable.
pub trait QuantumEstimator<T, V, A, M, O>
where
    A: ?Sized,
    M: ?Sized,
    O: ValidOutput<Self::Output>,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of a group to the contribution
    /// of the image to the observable.
    ///
    /// Where applicable, returns the observable.
    #[heavy_computation]
    fn calculate(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        positions: &GroupInTypeInImage<V>,
        physical_forces: &GroupInTypeInImage<V>,
        exchange_forces: &GroupInTypeInImage<V>,
    ) -> Result<O, Self::Error>;
}
