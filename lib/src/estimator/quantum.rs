//! Traits for calculating quantum observables.

use std::{
    error::Error,
    ops::{Deref, DerefMut},
};

use crate::core::{GroupInTypeInImage, Synchronizer};
use macros::heavy_computation;

mod atom_additive;
pub use atom_additive::{AdditiveValueQuantumEstimator, AdditiveVectorQuantumEstimator, AtomAdditiveQuantumEstimator};

mod atom_multiplicative;
pub use atom_multiplicative::{AtomMultiplicativeQuantumEstimator, MultiplicativeValueQuantumEstimator};

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
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Calculates the contribution of a group to the contribution
    /// of the image to the observable.
    ///
    /// Where applicable, returns the observable.
    #[heavy_computation]
    fn calculate(
        &mut self,
        synchronizer: &Synchronizer<T>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        positions: &GroupInTypeInImage<T, V>,
        physical_forces: &GroupInTypeInImage<T, V>,
        exchange_forces: &GroupInTypeInImage<T, V>,
    ) -> Result<O, Self::Error>;
}

impl<T, V, A, M, O, E> QuantumEstimator<T, V, A, M, O> for E
where
    A: ?Sized,
    M: ?Sized,
    E: DerefMut<Target: QuantumEstimator<T, V, A, M, O>> + ?Sized,
{
    type Output = <<Self as Deref>::Target as QuantumEstimator<T, V, A, M, O>>::Output;
    type Error = <<Self as Deref>::Target as QuantumEstimator<T, V, A, M, O>>::Error;

    #[inline]
    fn calculate(
        &mut self,
        synchronizer: &Synchronizer<T>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        positions: &GroupInTypeInImage<T, V>,
        physical_forces: &GroupInTypeInImage<T, V>,
        exchange_forces: &GroupInTypeInImage<T, V>,
    ) -> Result<O, Self::Error> {
        (*self).calculate(
            synchronizer,
            adder,
            multiplier,
            physical_potential_energy,
            type_exchange_potential_energy,
            positions,
            physical_forces,
            exchange_forces,
        )
    }
}
