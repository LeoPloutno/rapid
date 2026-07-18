//! Traits for updating the forces and calculating the physical potential energy.

use crate::core::{GroupInTypeInImage, ValidOutput};
use macros::{efficient_alternatives, heavy_computation};
use std::sync::{Barrier, RwLock};

mod atom_additive;
pub use atom_additive::AtomAdditivePhysicalPotential;

#[cfg(feature = "monte_carlo")]
mod monte_carlo;

#[cfg(feature = "monte_carlo")]
pub use self::{
    atom_additive::AtomAdditiveMonteCarloPhysicalPotential,
    monte_carlo::MonteCarloPhysicalPotential,
};

/// A trait for physical potentials.
///
/// The generic parameter `O` is the type of the values returned by energy calculations.
/// Setting it to `()` implies that the calculations are sent to another potential
/// that combines the recieved data and returns the total physical potential energy.
pub trait PhysicalPotential<T, V, A, M, O>
where
    A: ?Sized,
    M: ?Sized,
    O: ValidOutput<T>,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of a group to the total physical potential energy
    /// of the image and sets the forces of that group accordingly.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_energy_set_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error>;

    /// Calculates the contribution of a group to the total physical potential energy
    /// of the image and adds the forces arising from this potential to the forces of that group.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_energy_add_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error>;

    /// Calculates the contribution of a group to the total physical potential energy
    /// of the image.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    #[efficient_alternatives("calculate_energy_set_forces", "calculate_energy_set_forces")]
    fn calculate_energy(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<O, Self::Error>;

    /// Sets the forces of a group.
    #[efficient_alternatives("calculate_energy_set_forces")]
    fn set_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Adds the forces arising from this potential to the forces of a group.
    #[efficient_alternatives("calculate_energy_add_forces")]
    fn add_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error>;
}
