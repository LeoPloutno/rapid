use crate::{core::AtomGroupInfo, vector::Vector};

mod atom_decoupled;
mod group_decoupled;

/// A trait for physical potentials that yield the contribution of a single group
/// in a given replica to the total potential energy of said replica.
pub trait PhysicalPotential<T, const N: usize, V>
where
    V: Vector<N, Element = T>,
{
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
}

impl<T, const N: usize, V, U> PhysicalPotential<T, N, V> for U
where
    V: Vector<N, Element = T>,
    U: ?Sized + GroupDecoupledPhysicalPotential<T, N, V>,
{
    fn calculate_potential_set_forces(
        &mut self,
        _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        self.calculate_group_potential_set_forces(group, positions, forces)
    }

    fn calculate_potential(
        &mut self,
        _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        self.calculate_group_potential(group, positions)
    }

    fn set_forces(
        &mut self,
        _groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) {
        #[allow(deprecated)]
        self.set_group_forces(group, positions, forces);
    }
}

#[cfg(feature = "monte_carlo")]
mod monte_carlo {
    use super::{MonteCarloGroupDecoupledPhysicalPotential, PhysicalPotential};
    use crate::{core::AtomGroupInfo, vector::Vector};

    /// A trait for physical potentials that may be used in a Monte-Carlo algorithm.
    pub trait MonteCarloPhysicalPotential<T, const N: usize, V>:
        PhysicalPotential<T, N, V>
    where
        V: Vector<N, Element = T>,
    {
        /// Calculates the contribution of this group to the change in total potential energy
        /// of the replica after a change in the position of a single atom
        ///  and updates the forces of this group accordingly.
        ///
        /// Returns the contribution to the change in total energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff_update_forces(
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
        #[deprecated = "Consider using `calculate_potential_diff_set_forces` as a more efficient alternative"]
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
        #[deprecated = "Consider using `calculate_potential_diff_set_forces` as a more efficient alternative"]
        fn update_forces(
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

    impl<T, const N: usize, V, U> MonteCarloPhysicalPotential<T, N, V> for U
    where
        T: Default,
        V: Vector<N, Element = T>,
        U: ?Sized + MonteCarloGroupDecoupledPhysicalPotential<T, N, V>,
    {
        fn calculate_potential_diff_update_forces(
            &mut self,
            changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T {
            if groups_positions
                .nth(changed_group_idx)
                .expect("`changed_group_idx` must be a valid index in `groups_positions`")
                .0
                .id
                == group.id
            {
                self.calculate_group_potential_diff_update_force(
                    changed_position_idx,
                    old_value,
                    group,
                    positions,
                    forces,
                )
            } else {
                T::default()
            }
        }

        fn calculate_potential_diff(
            &mut self,
            changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<T>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
        ) -> T {
            if groups_positions
                .nth(changed_group_idx)
                .expect("`changed_group_idx` must be a valid index in `groups_positions`")
                .0
                .id
                == group.id
            {
                #[allow(deprecated)]
                self.calculate_group_potential_diff(
                    changed_position_idx,
                    old_value,
                    group,
                    positions,
                )
            } else {
                T::default()
            }
        }

        fn update_forces(
            &mut self,
            changed_group_idx: usize,
            changed_position_idx: usize,
            old_value: &V,
            groups_positions: &mut dyn Iterator<Item = (&AtomGroupInfo<V>, &[V])>,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) {
            if groups_positions
                .nth(changed_group_idx)
                .expect("`changed_group_idx` must be a valid index in `groups_positions`")
                .0
                .id
                == group.id
            {
                #[allow(deprecated)]
                self.update_group_forces(changed_position_idx, old_value, group, positions, forces)
            }
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
