//! Traits for updating the forces and calculating the physical potential energy.

use crate::core::GroupTypeHandle;
use macros::{efficient_alternatives, heavy_computation};

mod atom_additive;
pub use atom_additive::AtomAdditivePhysicalPotential;

#[cfg(feature = "monte_carlo")]
mod monte_carlo;
#[cfg(feature = "monte_carlo")]
pub use self::{atom_additive::AtomAdditiveMonteCarloPhysicalPotential, monte_carlo::MonteCarloPhysicalPotential};

/// A trait for physical potentials.
pub trait PhysicalPotential<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the image and sets the forces of this group accordingly.
    ///
    /// Returns the contribution to the total physical potential energy.
    #[heavy_computation]
    fn calculate_potential_set_forces(
        &mut self,
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the image and adds the forces arising from this potential to the forces of this group.
    ///
    /// Returns the contribution to the total physical potential energy.
    #[heavy_computation]
    fn calculate_potential_add_forces(
        &mut self,
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the image.
    ///
    /// Returns the contribution to the total physical potential energy.
    #[heavy_computation]
    #[efficient_alternatives("calculate_potential_set_forces", "calculate_potential_add_forces")]
    fn calculate_potential(&mut self, groups_positions: &[GroupTypeHandle<V>]) -> Result<T, Self::Error>;

    /// Sets the forces of this group.
    #[efficient_alternatives("calculate_potential_set_forces")]
    fn set_forces(
        &mut self,
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Adds the forces arising from this potential to the forces of this group.
    #[efficient_alternatives("calculate_potential_add_forces")]
    fn add_forces(
        &mut self,
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;
}
