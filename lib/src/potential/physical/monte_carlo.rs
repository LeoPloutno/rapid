use super::PhysicalPotential;
use crate::ImageHandle;

/// A trait for physical potentials that may be used in a Monte-Carlo algorithm.
pub trait MonteCarloPhysicalPotential<T, V>: PhysicalPotential<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and updates the group_forces of this group accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and adds the updated group_forces to the group_forces of this group.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom.
    ///
    /// Returns the contribution to the change in total energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error>;

    /// Updates the group_forces of this group after a change in the position of a single atom.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    fn set_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V>>::Error>;

    /// Adds the updated group_forces to the group_forces of this group given a change
    /// in the position of a single atom.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    fn add_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V>>::Error>;
}
