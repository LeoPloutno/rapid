//! Traits for updating the forces and calculating the physical potential energy.

use crate::ImageHandle;

mod atom_additive;
#[cfg(feature = "monte_carlo")]
mod monte_carlo;

#[cfg(feature = "monte_carlo")]
pub use self::{atom_additive::AtomAdditiveMonteCarloPhysicalPotential, monte_carlo::MonteCarloPhysicalPotential};
pub use atom_additive::AtomAdditivePhysicalPotential;

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
