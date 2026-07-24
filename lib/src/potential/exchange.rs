//! Traits for updating the forces and calculating the exchange potential energy.

use crate::core::{AtomGroup, GroupInTypeInImage, ValidOutput};
use macros::{efficient_alternatives, heavy_computation};
use std::sync::{Barrier, RwLock};

pub mod quadratic;

#[cfg(feature = "monte_carlo")]
mod monte_carlo;
#[cfg(feature = "monte_carlo")]
pub use monte_carlo::{MonteCarloExchangePotential, NeighboringImage};

/// A trait for exchange potentials.
///
/// The generic parameter `O` is the type of the values returned by the energy calculations.
/// Setting it to `()` implies that the calculations are sent to another potential
/// that combines the recieved data and returns the total exchange potential energy.
pub trait ExchangePotential<T, V, A, M, O>
where
    A: ?Sized,
    M: ?Sized,
    O: ValidOutput<T>,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of a group to the total exchange potential energy
    /// of the type and sets the forces of that group accordingly.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_energy_set_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error>;

    /// Calculates the contribution of a group to the total exchange potential energy
    /// of the type and adds the forces arising from this potential to the forces of that group.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_energy_add_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error>;

    /// Calculates the contribution of a group to the total exchange potential energy
    /// of the type.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    #[efficient_alternatives("calculate_energy_set_forces", "calculate_energy_add_forces")]
    fn calculate_energy(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
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
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
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
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error>;
}
