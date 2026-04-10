use super::PhysicalPotential;
use crate::core::GroupTypeHandle;
use macros::{efficient_alternatives, heavy_computation};

/// A trait for physical potentials that may be used in a Monte-Carlo algorithm.
pub trait MonteCarloPhysicalPotential<T, V>: PhysicalPotential<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and sets the forces of this group accordingly.
    ///
    /// Returns the contribution to the change in total physical potential energy.
    #[heavy_computation]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and adds the forces arising from this potential to the forces of this group.
    ///
    /// Returns the contribution to the change in total physical potential energy.
    #[heavy_computation]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom.
    ///
    /// Returns the contribution to the change in total physical potential energy.
    #[heavy_computation]
    #[efficient_alternatives(
        "calculate_potential_diff_set_changed_forces",
        "calculate_potential_diff_add_changed_forces"
    )]
    fn calculate_potential_diff(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &[GroupTypeHandle<V>],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error>;

    /// Sets the forces of this group after a change in the position of a single atom.
    #[efficient_alternatives("calculate_potential_diff_set_changed_forces")]
    fn set_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V>>::Error>;

    /// Adds the forces arising from this potential to the forces of this group
    /// after a change in the position of a single atom.
    #[efficient_alternatives("calculate_potential_diff_add_changed_forces")]
    fn add_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V>>::Error>;
}
