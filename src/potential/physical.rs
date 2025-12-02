use crate::core::AtomGroupInfo;

mod atom_decoupled;
mod group_decoupled;

/// A trait for physical potentials that yield the contribution of a single group
/// in a given replica to the total potential energy of said replica.
pub trait PhysicalPotential<T, V> {
    /// Calculates the contribution of this group to the total potential energy
    /// of the replica and sets the forces of this group accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the total potential energy
    /// of the replica and adds the forces arising from this potential to the forces of this group.
    ///
    /// Returns the contribution to the total energy.
    fn calculate_potential_add_forces(
        &mut self,
        groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the total potential energy
    /// of the replica.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
    ) -> T;

    /// Sets the forces of this group.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    );

    /// Adds the forces arising from this potential to the forces of this group.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    );
}

impl<T, V, U> PhysicalPotential<T, V> for U
where
    U: ?Sized + GroupDecoupledPhysicalPotential<T, V>,
{
    fn calculate_potential_set_forces(
        &mut self,
        _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        GroupDecoupledPhysicalPotential::calculate_potential_set_forces(
            self, group, positions, forces,
        )
    }

    fn calculate_potential_add_forces(
        &mut self,
        _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        GroupDecoupledPhysicalPotential::calculate_potential_add_forces(
            self, group, positions, forces,
        )
    }

    fn calculate_potential(
        &mut self,
        _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        GroupDecoupledPhysicalPotential::calculate_potential(self, group, positions)
    }

    fn set_forces(
        &mut self,
        _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) {
        #[allow(deprecated)]
        GroupDecoupledPhysicalPotential::set_forces(self, group, positions, forces);
    }

    fn add_forces(
        &mut self,
        _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) {
        #[allow(deprecated)]
        GroupDecoupledPhysicalPotential::add_forces(self, group, positions, forces);
    }
}

#[cfg(feature = "monte_carlo")]
mod monte_carlo {
    use super::{MonteCarloGroupDecoupledPhysicalPotential, PhysicalPotential};
    use crate::core::AtomGroupInfo;

    /// A trait for physical potentials that may be used in a Monte-Carlo algorithm.
    pub trait MonteCarloPhysicalPotential<T, V>: PhysicalPotential<T, V> {
        /// Calculates the contribution of this group to the change in total potential energy
        /// of the replica after a change in the position of a single atom
        /// and updates the forces of this group accordingly.
        ///
        /// Returns the contribution to the change in total energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff_set_changed_forces(
            &mut self,
            changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T;

        /// Calculates the contribution of this group to the change in total potential energy
        /// of the replica after a change in the position of a single atom
        /// and adds the updated forces to the forces of this group.
        ///
        /// Returns the contribution to the change in total energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff_add_changed_forces(
            &mut self,
            changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T;

        /// Calculates the contribution of this group to the change in total potential energy
        /// of the replica after a change in the position of a single atom.
        ///
        /// Returns the contribution to the change in total energy.
        #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff(
            &mut self,
            changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
        ) -> T;

        /// Updates the forces of this group after a change in the position of a single atom.
        #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
        fn set_changed_forces(
            &mut self,
            changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        );

        /// Adds the updated forces to the forces of this group given a change
        /// in the position of a single atom.
        #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
        fn add_changed_forces(
            &mut self,
            changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        );
    }

    impl<T, V, U> MonteCarloPhysicalPotential<T, V> for U
    where
        U: ?Sized + MonteCarloGroupDecoupledPhysicalPotential<T, V>,
    {
        fn calculate_potential_diff_set_changed_forces(
            &mut self,
            _changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T {
            MonteCarloGroupDecoupledPhysicalPotential::calculate_potential_diff_set_changed_forces(
                self,
                changed_position_idx,
                old_value,
                group,
                positions,
                forces,
            )
        }

        fn calculate_potential_diff_add_changed_forces(
            &mut self,
            _changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T {
            MonteCarloGroupDecoupledPhysicalPotential::calculate_potential_diff_add_changed_forces(
                self,
                changed_position_idx,
                old_value,
                group,
                positions,
                forces,
            )
        }

        fn calculate_potential_diff(
            &mut self,
            _changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
        ) -> T {
            #[allow(deprecated)]
            MonteCarloGroupDecoupledPhysicalPotential::calculate_potential_diff(
                self,
                changed_position_idx,
                old_value,
                group,
                positions,
            )
        }

        fn set_changed_forces(
            &mut self,
            _changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) {
            #[allow(deprecated)]
            MonteCarloGroupDecoupledPhysicalPotential::set_changed_forces(
                self,
                changed_position_idx,
                old_value,
                group,
                positions,
                forces,
            );
        }

        fn add_changed_forces(
            &mut self,
            _changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) {
            #[allow(deprecated)]
            MonteCarloGroupDecoupledPhysicalPotential::add_changed_forces(
                self,
                changed_position_idx,
                old_value,
                group,
                positions,
                forces,
            );
        }
    }
}

pub use self::{
    atom_decoupled::AtomDecoupledPhysicalPotential,
    group_decoupled::GroupDecoupledPhysicalPotential,
};
#[cfg(feature = "monte_carlo")]
pub use self::{
    atom_decoupled::monte_carlo::MonteCarloAtomDecoupledPhysicalPotential,
    group_decoupled::monte_carlo::MonteCarloGroupDecoupledPhysicalPotential,
    monte_carlo::MonteCarloPhysicalPotential,
};
