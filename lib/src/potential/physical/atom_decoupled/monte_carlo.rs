use super::AtomDecoupledPhysicalPotential;
use crate::core::AtomGroupInfo;

/// A trait for atom-deoupled physical potentials that may be used in a Monte-Carlo algorithm.
///
/// Any implementor of this trait automatically implements [`MonteCarloGroupDecoupledPhysicalPotential`].
///
/// [`MonteCarloGroupDecoupledPhysicalPotential`]: super::super::MonteCarloGroupDecoupledPhysicalPotential
pub trait MonteCarloAtomDecoupledPhysicalPotential<T, V>:
    AtomDecoupledPhysicalPotential<T, V>
{
    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its atom_position and updates the atom_force of this atom accordingly.
    ///
    /// Returns the change in potential energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_force(
        &mut self,
        group: &AtomGroupInfo<T>,
        atom_idx: usize,
        old_value: &V,
        atom_position: &V,
        atom_force: &mut V,
    ) -> T;

    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its atom_position and adds the updated atom_force to the atom_force
    /// of this atom.
    ///
    /// Returns the change in potential energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_force(
        &mut self,
        group: &AtomGroupInfo<T>,
        atom_idx: usize,
        old_value: &V,
        atom_position: &V,
        atom_force: &mut V,
    ) -> T;

    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its atom_position.
    ///
    /// Returns the change in potential energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_force` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        atom_position: &V,
    ) -> T;

    /// Updates the atom_force of this atom after a change to its atom_position.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_force` as a more efficient alternative"]
    fn set_changed_force(
        &mut self,
        group: &AtomGroupInfo<T>,
        atom_idx: usize,
        old_value: &V,
        atom_position: &V,
        atom_force: &mut V,
    );

    /// Adds the updated atom_force to the atom_force of this atom given a change in its atom_position.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_force` as a more efficient alternative"]
    fn add_changed_force(
        &mut self,
        group: &AtomGroupInfo<T>,
        atom_idx: usize,
        old_value: &V,
        atom_position: &V,
        atom_force: &mut V,
    );
}
