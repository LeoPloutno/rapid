//! Traits for updating the forces and calculating the physical potential energy.

use std::ops::Add;

use crate::{ImageHandle, core::error::EmptyIteratorError};

mod atom_decoupled;
#[cfg(feature = "monte_carlo")]
mod monte_carlo;

/// A trait for physical potentials that yield the contribution of a single group
/// in a given image to the total physical potential energy of said image.
pub trait PhysicalPotential<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group to the total potential energy
    /// of the image and sets the forces of this group accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group to the total potential energy
    /// of the image and adds the forces arising from this potential to the forces of this group.
    ///
    /// Returns the contribution to the total energy.
    fn calculate_potential_add_forces(
        &mut self,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group to the total potential energy
    /// of the image.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(&mut self, groups_positions: &ImageHandle<V>) -> Result<T, Self::Error>;

    /// Sets the forces of this group.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(&mut self, groups_positions: &ImageHandle<V>, group_forces: &mut [V]) -> Result<(), Self::Error>;

    /// Adds the forces arising from this potential to the forces of this group.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(&mut self, groups_positions: &ImageHandle<V>, group_forces: &mut [V]) -> Result<(), Self::Error>;
}

impl<T, V, U> PhysicalPotential<T, V> for U
where
    T: Add<Output = T>,
    U: AtomDecoupledPhysicalPotential<T, V, Error: From<EmptyIteratorError>> + ?Sized,
{
    type Error = <Self as AtomDecoupledPhysicalPotential<T, V>>::Error;

    fn calculate_potential_set_forces(
        &mut self,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        let mut iter = groups_positions
            .read()
            .read()
            .iter()
            .zip(group_forces.iter_mut())
            .enumerate()
            .map(|(index, (position, force))| {
                AtomDecoupledPhysicalPotential::calculate_potential_set_force(self, index, position, force)
            });
        let first_atom_energy = iter.next().ok_or(EmptyIteratorError)??;
        iter.try_fold(first_atom_energy, |accum_energy, atom_energy| {
            Ok(accum_energy + atom_energy?)
        })
    }

    fn calculate_potential_add_forces(
        &mut self,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        let mut iter = groups_positions
            .read()
            .read()
            .iter()
            .zip(group_forces.iter_mut())
            .enumerate()
            .map(|(index, (position, force))| {
                AtomDecoupledPhysicalPotential::calculate_potential_add_force(self, index, position, force)
            });
        let first_atom_energy = iter.next().ok_or(EmptyIteratorError)??;
        iter.try_fold(first_atom_energy, |accum_energy, atom_energy| {
            Ok(accum_energy + atom_energy?)
        })
    }

    fn calculate_potential(&mut self, groups_positions: &ImageHandle<V>) -> Result<T, Self::Error> {
        let mut iter = groups_positions
            .read()
            .read()
            .iter()
            .enumerate()
            .map(|(index, position)| {
                #[allow(deprecated)]
                AtomDecoupledPhysicalPotential::calculate_potential(self, index, position)
            });
        let first_atom_energy = iter.next().ok_or(EmptyIteratorError)??;
        iter.try_fold(first_atom_energy, |accum_energy, atom_energy| {
            Ok(accum_energy + atom_energy?)
        })
    }

    fn set_forces(&mut self, groups_positions: &ImageHandle<V>, group_forces: &mut [V]) -> Result<(), Self::Error> {
        for (index, (position, force)) in groups_positions
            .read()
            .read()
            .iter()
            .zip(group_forces.iter_mut())
            .enumerate()
        {
            #[allow(deprecated)]
            AtomDecoupledPhysicalPotential::set_force(self, index, position, force)?;
        }
        Ok(())
    }

    fn add_forces(&mut self, groups_positions: &ImageHandle<V>, group_forces: &mut [V]) -> Result<(), Self::Error> {
        for (index, (position, force)) in groups_positions
            .read()
            .read()
            .iter()
            .zip(group_forces.iter_mut())
            .enumerate()
        {
            #[allow(deprecated)]
            AtomDecoupledPhysicalPotential::add_force(self, index, position, force)?;
        }
        Ok(())
    }
}

#[cfg(feature = "monte_carlo")]
pub use self::{atom_decoupled::MonteCarloAtomDecoupledPhysicalPotential, monte_carlo::MonteCarloPhysicalPotential};
pub use atom_decoupled::AtomDecoupledPhysicalPotential;
