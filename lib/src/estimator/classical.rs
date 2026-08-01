//! Traits for calculating classical observables.

use crate::core::{GroupInTypeInImageInSystem, marker::ValidOutput};
use macros::heavy_computation;
use std::sync::{Barrier, RwLock};

mod atom_additive;
pub use atom_additive::{
    AdditiveValueClassicalEstimator, AdditiveVectorClassicalEstimator,
    AtomAdditiveClassicalEstimator,
};

mod atom_multiplicative;
pub use atom_multiplicative::{
    AtomMultiplicativeClassicalEstimator, MultiplicativeValueClassicalEstimator,
};

/// A trait for classical estimators.
///
/// The generic parameter `O` is the type of the values returned by calculations.
/// Setting it to `()` implies that the calculations are sent to another estimator
/// that combines the recieved data and returns the final observable.
pub trait ClassicalEstimator<T, V, A, M, O>
where
    A: ?Sized,
    M: ?Sized,
    O: ValidOutput<Self::Output>,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of a group to the observable.
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
        group_kinetic_energy: T,
        group_heat: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<O, Self::Error>;
}
