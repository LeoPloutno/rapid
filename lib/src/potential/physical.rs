//! Traits for updating the forces and calculating the physical potential energy.

use std::{
    error::Error,
    ops::{Deref, DerefMut},
};

use crate::core::{GroupInTypeInImage, marker::ValidOutput};
use macros::{efficient_alternatives, heavy_computation};

mod atom_additive;
pub use atom_additive::AtomAdditivePhysicalPotential;

#[cfg(feature = "monte_carlo")]
mod monte_carlo;

#[cfg(feature = "monte_carlo")]
pub use self::{
    atom_additive::{AdditivePhysicalPotential, AtomAdditiveMonteCarloPhysicalPotential},
    monte_carlo::{ChangedGroup, MonteCarloPhysicalPotential},
};

/// A trait for physical potentials.
///
/// The generic parameter `O` is the type of the values returned by the energy calculations.
/// Setting it to `()` implies that the calculations are sent to another potential
/// that combines the recieved data and returns the total physical potential energy.
pub trait PhysicalPotential<T, V, O: ValidOutput<T>> {
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Calculates the contribution of a group to the total physical potential energy
    /// of the image and sets the forces of that group accordingly.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_energy_set_forces(
        &mut self,
        positions: &GroupInTypeInImage<T, V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error>;

    /// Calculates the contribution of a group to the total physical potential energy
    /// of the image and adds the forces arising from this potential to the forces of that group.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_energy_add_forces(
        &mut self,
        positions: &GroupInTypeInImage<T, V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error>;

    /// Calculates the contribution of a group to the total physical potential energy
    /// of the image.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    #[efficient_alternatives("calculate_energy_set_forces", "calculate_energy_add_forces")]
    fn calculate_energy(&mut self, positions: &GroupInTypeInImage<T, V>) -> Result<O, Self::Error>;

    /// Sets the forces of a group.
    #[efficient_alternatives("calculate_energy_set_forces")]
    fn set_forces(&mut self, positions: &GroupInTypeInImage<T, V>, forces: &mut [V]) -> Result<(), Self::Error>;

    /// Adds the forces arising from this potential to the forces of a group.
    #[efficient_alternatives("calculate_energy_add_forces")]
    fn add_forces(&mut self, positions: &GroupInTypeInImage<T, V>, forces: &mut [V]) -> Result<(), Self::Error>;
}

impl<T, V, O, P> PhysicalPotential<T, V, O> for P
where
    O: ValidOutput<T>,
    P: DerefMut<Target: PhysicalPotential<T, V, O>> + ?Sized,
{
    type Error = <<Self as Deref>::Target as PhysicalPotential<T, V, O>>::Error;

    #[inline]
    fn calculate_energy_set_forces(
        &mut self,
        positions: &GroupInTypeInImage<T, V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error> {
        <<Self as Deref>::Target as PhysicalPotential<T, V, O>>::calculate_energy_set_forces(
            &mut *self, positions, forces,
        )
    }

    #[inline]
    fn calculate_energy_add_forces(
        &mut self,
        positions: &GroupInTypeInImage<T, V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error> {
        <<Self as Deref>::Target as PhysicalPotential<T, V, O>>::calculate_energy_add_forces(
            &mut *self, positions, forces,
        )
    }

    #[inline]
    fn calculate_energy(&mut self, positions: &GroupInTypeInImage<T, V>) -> Result<O, Self::Error> {
        #[allow(deprecated)]
        (*self).calculate_energy(positions)
    }

    #[inline]
    fn set_forces(&mut self, positions: &GroupInTypeInImage<T, V>, forces: &mut [V]) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        <<Self as Deref>::Target as PhysicalPotential<T, V, O>>::set_forces(&mut *self, positions, forces)
    }

    #[inline]
    fn add_forces(&mut self, positions: &GroupInTypeInImage<T, V>, forces: &mut [V]) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        <<Self as Deref>::Target as PhysicalPotential<T, V, O>>::add_forces(&mut *self, positions, forces)
    }
}
