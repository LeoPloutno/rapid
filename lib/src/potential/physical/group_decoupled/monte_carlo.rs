use std::ops::Add;

use super::GroupDecoupledPhysicalPotential;
use crate::{core::AtomGroupInfo, potential::physical::MonteCarloAtomDecoupledPhysicalPotential};

/// A trait for group-deoupled physical potentials that may be used in a Monte-Carlo algorithm.
///
/// Any implementor of this trait automatically implements [`MonteCarloPhysicalPotential`]
///
/// [`MonteCarloPhysicalPotential`]: super::super::MonteCarloPhysicalPotential
pub trait MonteCarloGroupDecoupledPhysicalPotential<T, V>:
    GroupDecoupledPhysicalPotential<T, V>
{
    /// Calculates the change in the physical potential energy of this group
    /// after a change in the position of one of its atoms
    /// and sets the group_forces of this group accordingly.
    ///
    /// Returns the change in potential energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the change in the physical potential energy of this group
    /// after a change in the position of one of its atoms
    /// and adds the updated group_forces to the group_forces of this group.
    ///
    /// Returns the change in potential energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the change in the physical potential energy of this group
    /// after a change in the position of one of its atoms.
    ///
    /// Returns the change in potential energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
    ) -> T;

    /// Updates the group_forces of this group after a change in the position of one of its atoms.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    fn set_changed_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
        group_forces: &mut [V],
    );

    /// Adds the updated group_forces to the group_forces of this group given a change
    /// in the position of one of its atoms.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    fn add_changed_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
        group_forces: &mut [V],
    );
}

impl<T, V, U> MonteCarloGroupDecoupledPhysicalPotential<T, V> for U
where
    T: Add<Output = T>,
    U: ?Sized + MonteCarloAtomDecoupledPhysicalPotential<T, V>,
{
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        MonteCarloAtomDecoupledPhysicalPotential::calculate_potential_diff_set_changed_force(
            self,
            group,
            changed_position_idx,
            old_value,
            group_positions
                .get(changed_position_idx)
                .expect("`changed_position_idx` must be a valid index in `group_positions`"),
            group_forces
                .get_mut(changed_position_idx)
                .expect("`changed_position_idx` must be a valid index in `group_forces`"),
        )
    }

    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        MonteCarloAtomDecoupledPhysicalPotential::calculate_potential_diff_add_changed_force(
            self,
            group,
            changed_position_idx,
            old_value,
            group_positions
                .get(changed_position_idx)
                .expect("`changed_position_idx` must be a valid index in `group_positions`"),
            group_forces
                .get_mut(changed_position_idx)
                .expect("`changed_position_idx` must be a valid index in `group_forces`"),
        )
    }

    fn calculate_potential_diff(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        MonteCarloAtomDecoupledPhysicalPotential::calculate_potential_diff(
            self,
            group,
            changed_position_idx,
            old_value,
            group_positions
                .get(changed_position_idx)
                .expect("`changed_position_idx` must be a valid index in `group_positions`"),
        )
    }

    fn set_changed_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        MonteCarloAtomDecoupledPhysicalPotential::set_changed_force(
            self,
            group,
            changed_position_idx,
            old_value,
            group_positions
                .get(changed_position_idx)
                .expect("`changed_position_idx` must be a valid index in `group_positions`"),
            group_forces
                .get_mut(changed_position_idx)
                .expect("`changed_position_idx` must be a valid index in `group_forces`"),
        );
    }

    fn add_changed_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        changed_position_idx: usize,
        old_value: &V,
        group_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        MonteCarloAtomDecoupledPhysicalPotential::add_changed_force(
            self,
            group,
            changed_position_idx,
            old_value,
            group_positions
                .get(changed_position_idx)
                .expect("`changed_position_idx` must be a valid index in `group_positions`"),
            group_forces
                .get_mut(changed_position_idx)
                .expect("`changed_position_idx` must be a valid index in `group_forces`"),
        );
    }
}
