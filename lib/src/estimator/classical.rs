//! Traits for calculating classical observables.

use std::{
    error::Error,
    ops::{Deref, DerefMut},
};

use crate::core::{GroupInTypeInImageInSystem, Synchronizer};
use macros::heavy_computation;

mod atom_additive;
pub use atom_additive::{
    AdditiveValueClassicalEstimator, AdditiveVectorClassicalEstimator, AtomAdditiveClassicalEstimator,
};

mod atom_multiplicative;
pub use atom_multiplicative::{AtomMultiplicativeClassicalEstimator, MultiplicativeValueClassicalEstimator};

/// A trait for classical estimators.
///
/// The generic parameter `O` is the type of the values returned by calculations.
/// Setting it to `()` implies that the calculations are sent to another estimator
/// that combines the recieved data and returns the final observable.
pub trait ClassicalEstimator<T, V, A, M, O>
where
    A: ?Sized,
    M: ?Sized,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Calculates the contribution of a group to the observable.
    ///
    /// Where applicable, returns the observable.
    #[heavy_computation]
    fn calculate(
        &mut self,
        system_synchronizer: &Synchronizer<T>,
        image_synchronizer: &Synchronizer<T>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        group_heat: T,
        positions: &GroupInTypeInImageInSystem<T, V>,
        momenta: &GroupInTypeInImageInSystem<T, V>,
        physical_forces: &GroupInTypeInImageInSystem<T, V>,
        exchange_forces: &GroupInTypeInImageInSystem<T, V>,
    ) -> Result<O, Self::Error>;
}

impl<T, V, A, M, O, E> ClassicalEstimator<T, V, A, M, O> for E
where
    A: ?Sized,
    M: ?Sized,
    E: DerefMut<Target: ClassicalEstimator<T, V, A, M, O>> + ?Sized,
{
    type Output = <<Self as Deref>::Target as ClassicalEstimator<T, V, A, M, O>>::Output;
    type Error = <<Self as Deref>::Target as ClassicalEstimator<T, V, A, M, O>>::Error;

    #[inline]
    fn calculate(
        &mut self,
        system_synchronizer: &Synchronizer<T>,
        image_synchronizer: &Synchronizer<T>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        group_heat: T,
        positions: &GroupInTypeInImageInSystem<T, V>,
        momenta: &GroupInTypeInImageInSystem<T, V>,
        physical_forces: &GroupInTypeInImageInSystem<T, V>,
        exchange_forces: &GroupInTypeInImageInSystem<T, V>,
    ) -> Result<O, Self::Error> {
        (*self).calculate(
            system_synchronizer,
            image_synchronizer,
            adder,
            multiplier,
            physical_potential_energy,
            type_exchange_potential_energy,
            group_heat,
            positions,
            momenta,
            physical_forces,
            exchange_forces,
        )
    }
}
