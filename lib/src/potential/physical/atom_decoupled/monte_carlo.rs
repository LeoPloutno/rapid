use super::AtomDecoupledPhysicalPotential;
use crate::core::AtomGroupInfo;

/// A trait for atom-deoupled physical potentials that may be used in a Monte-Carlo algorithm.
///
/// Any implementor of this trait automatically implements [`MonteCarloGroupDecoupledPhysicalPotential`].
///
/// [`MonteCarloGroupDecoupledPhysicalPotential`]: super::super::MonteCarloGroupDecoupledPhysicalPotential
pub trait MonteCarloAtomDecoupledPhysicalPotential<T, V>: AtomDecoupledPhysicalPotential<T, V> {
    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its position and updates the force of this atom accordingly.
    ///
    /// Returns the change in potential energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_force(
        &mut self,
        atom_idx: usize,
        group: &AtomGroupInfo<T>,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> T;

    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its position and adds the updated force to the force
    /// of this atom.
    ///
    /// Returns the change in potential energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_force(
        &mut self,
        atom_idx: usize,
        group: &AtomGroupInfo<T>,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> T;

    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its position.
    ///
    /// Returns the change in potential energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_force` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(&mut self, atom_idx: usize, group: &AtomGroupInfo<T>, old_value: V, position: &V) -> T;

    /// Updates the force of this atom after a change to its position.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_force` as a more efficient alternative"]
    fn set_changed_force(
        &mut self,
        atom_idx: usize,
        group: &AtomGroupInfo<T>,
        old_value: V,
        position: &V,
        force: &mut V,
    );

    /// Adds the updated force to the force of this atom given a change in its position.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_force` as a more efficient alternative"]
    fn add_changed_force(
        &mut self,
        atom_idx: usize,
        group: &AtomGroupInfo<T>,
        old_value: V,
        position: &V,
        force: &mut V,
    );
}
