use super::{MonteCarloGroupDecoupledPhysicalPotential, PhysicalPotential};
use crate::core::GroupRecord;

/// A trait for physical potentials that may be used in a Monte-Carlo algorithm.
pub trait MonteCarloPhysicalPotential<T, V>: PhysicalPotential<T, V> {
    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the replica after a change in the position of a single atom
    /// and updates the group_forces of this group accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the replica after a change in the position of a single atom
    /// and adds the updated group_forces to the group_forces of this group.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the replica after a change in the position of a single atom.
    ///
    /// Returns the contribution to the change in total energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
    ) -> T;

    /// Updates the group_forces of this group after a change in the position of a single atom.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    fn set_changed_forces(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
        group_forces: &mut [V],
    );

    /// Adds the updated group_forces to the group_forces of this group given a change
    /// in the position of a single atom.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    fn add_changed_forces(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
        group_forces: &mut [V],
    );
}

impl<T, V, U> MonteCarloPhysicalPotential<T, V> for U
where
    T: Default,
    U: MonteCarloGroupDecoupledPhysicalPotential<T, V> + GroupRecord<T> + ?Sized,
{
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        if changed_group_idx == self.group_idx() {
            let group_positions = groups_positions
                .get(self.group().span)
                .expect("`span` should be a valid range in `groups_positions`");
            MonteCarloGroupDecoupledPhysicalPotential::calculate_potential_diff_set_changed_forces(
                self,
                changed_atom_idx,
                old_value,
                group_positions,
                group_forces,
            )
        } else {
            T::default()
        }
    }

    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        if changed_group_idx == self.group_idx() {
            let group_positions = groups_positions
                .get(self.group().span)
                .expect("`span` should be a valid range in `groups_positions`");
            MonteCarloGroupDecoupledPhysicalPotential::calculate_potential_diff_add_changed_forces(
                self,
                changed_atom_idx,
                old_value,
                group_positions,
                group_forces,
            )
        } else {
            T::default()
        }
    }

    fn calculate_potential_diff(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
    ) -> T {
        if changed_group_idx == self.group_idx() {
            let group_positions = groups_positions
                .get(self.group().span)
                .expect("`span` should be a valid range in `groups_positions`");
            #[allow(deprecated)]
            MonteCarloGroupDecoupledPhysicalPotential::calculate_potential_diff(
                self,
                changed_atom_idx,
                old_value,
                group_positions,
            )
        } else {
            T::default()
        }
    }

    fn set_changed_forces(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
        group_forces: &mut [V],
    ) {
        if changed_group_idx == self.group_idx() {
            let group_positions = groups_positions
                .get(self.group().span)
                .expect("`span` should be a valid range in `groups_positions`");
            #[allow(deprecated)]
            MonteCarloGroupDecoupledPhysicalPotential::set_changed_forces(
                self,
                changed_atom_idx,
                old_value,
                group_positions,
                group_forces,
            );
        }
    }

    fn add_changed_forces(
        &mut self,
        changed_group_idx: usize,
        changed_atom_idx: usize,
        old_value: V,
        groups_positions: &[V],
        group_forces: &mut [V],
    ) {
        if changed_group_idx == self.group_idx() {
            let group_positions = groups_positions
                .get(self.group().span)
                .expect("`span` should be a valid range in `groups_positions`");
            #[allow(deprecated)]
            MonteCarloGroupDecoupledPhysicalPotential::add_changed_forces(
                self,
                changed_atom_idx,
                old_value,
                group_positions,
                group_forces,
            );
        }
    }
}
