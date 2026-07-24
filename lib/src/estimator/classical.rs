//! Traits for calculating classical observables.

use crate::core::{GroupInTypeInImageInSystem, ValidOutput};

mod atom_additive;
pub use atom_additive::{
    AtomAdditiveClassicalEstimatorReceiver, AtomAdditiveClassicalEstimatorSender,
    AtomAdditiveMinimalClassicalEstimatorSender,
};
mod atom_multiplicative;
pub use atom_multiplicative::{
    AtomMultiplicativeClassicalEstimatorReceiver, AtomMultiplicativeClassicalEstimatorSender,
    AtomMultiplicativeMinimalClassicalEstimatorSender,
};

/// A trait for classical estimators.
///
/// The generic parameter `O` is the type of the values returned by the calculation.
/// Setting it to `()` implies that the calculations are sent to another observable
/// that combines the recieved data and returns the final value.
pub trait ClassicalEstimator<T, V, A, M, O>
where
    A: ?Sized,
    M: ?Sized,
    O: ValidOutput<T>,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of a group to the observable.
    ///
    /// Where applicable, returns the observable.
    fn calculate_distinguishable(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<O, Self::Error>;
}
