use std::ops::Add;

use crate::{core::AtomGroupInfo, vector::Vector};

use super::AtomDecoupledPhysicalPotential;

/// A trait for physical potentials that yield the contribution of a single group
/// in a given replica to the total potential energy of said replica.
pub trait GroupDecoupledPhysicalPotential<T, const N: usize, V>
where
    V: Vector<N, Element = T>,
{
    /// Calculates the contribution of this group to the total potential energy
    /// of the replica and sets the forces of this group accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_group_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the total potential energy
    /// of the replica.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_group_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_group_potential(&mut self, group: &AtomGroupInfo<T>, positions: &[V]) -> T;

    /// Sets the forces of this group.
    #[deprecated = "Consider using `calculate_group_potential_set_forces` as a more efficient alternative"]
    fn set_group_forces(&mut self, group: &AtomGroupInfo<T>, positions: &[V], forces: &mut [V]);
}

impl<T, const N: usize, V, U> GroupDecoupledPhysicalPotential<T, N, V> for [U]
where
    T: Add<Output = T>,
    V: Vector<N, Element = T>,
    U: AtomDecoupledPhysicalPotential<T, N, V>,
{
    fn calculate_group_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        let mut iter = positions
            .iter()
            .zip(forces.iter_mut())
            .zip(self)
            .enumerate()
            .map(|(idx, ((position, force), potential))| {
                potential.calculate_atom_potential_set_force(idx, group, position, force)
            });
        let first_atom_energy = iter
            .next()
            .expect("There must be at least one atom in a group");
        iter.fold(first_atom_energy, |accum, energy| accum + energy)
    }

    fn calculate_group_potential(&mut self, group: &AtomGroupInfo<T>, positions: &[V]) -> T {
        let mut iter =
            positions
                .iter()
                .zip(self)
                .enumerate()
                .map(|(idx, (position, potential))| {
                    #[allow(deprecated)]
                    potential.calculate_atom_potential(idx, group, position)
                });
        let first_atom_energy = iter
            .next()
            .expect("There must be at least one atom in a group");
        iter.fold(first_atom_energy, |accum, energy| accum + energy)
    }

    fn set_group_forces(&mut self, group: &AtomGroupInfo<T>, positions: &[V], forces: &mut [V]) {
        for (idx, ((position, force), potential)) in positions
            .iter()
            .zip(forces.iter_mut())
            .zip(self)
            .enumerate()
        {
            #[allow(deprecated)]
            potential.set_atom_force(idx, group, position, force);
        }
    }
}

#[cfg(feature = "monte_carlo")]
pub(super) mod monte_carlo {
    use std::ops::Add;

    use super::GroupDecoupledPhysicalPotential;
    use crate::{
        core::AtomGroupInfo, potential::physical::MonteCarloAtomDecoupledPhysicalPotential,
        vector::Vector,
    };

    /// A trait for group-deoupled physical potentials that may be used in a Monte-Carlo algorithm.
    pub trait MonteCarloGroupDecoupledPhysicalPotential<T, const N: usize, V>:
        GroupDecoupledPhysicalPotential<T, N, V>
    where
        V: Vector<N, Element = T>,
    {
        /// Calculates the change in the potential energy of this group
        /// after a change in the position of one of its atoms
        /// and sets the forces of this group accordingly.
        ///
        /// Returns the change in potential energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_group_potential_diff_update_force(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T;

        /// Calculates the change in the potential energy of this group
        /// after a change in the position of one of its atoms.
        ///
        /// Returns the change in potential energy.
        #[deprecated = "Consider using `calculate_group_potential_diff_update_force` as a more efficient alternative"]
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_group_potential_diff(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
        ) -> T;

        /// Updates the forces of this group after a change in the position of one of its atoms.
        #[deprecated = "Consider using `calculate_group_potential_diff_update_force` as a more efficient alternative"]
        fn update_group_forces(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        );
    }

    impl<T, const N: usize, V, U> MonteCarloGroupDecoupledPhysicalPotential<T, N, V> for [U]
    where
        T: Add<Output = T>,
        V: Vector<N, Element = T>,
        U: MonteCarloAtomDecoupledPhysicalPotential<T, N, V>,
    {
        fn calculate_group_potential_diff_update_force(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T {
            self.into_iter()
                .nth(changed_position_idx)
                .expect("There must be at least one atom-decoupled potential in the slice")
                .calculate_atom_potential_diff_update_force(
                    changed_position_idx,
                    old_value,
                    group,
                    positions
                        .get(changed_position_idx)
                        .expect("`changed_position_idx` must be a valid index in `positions`"),
                    forces
                        .get_mut(changed_position_idx)
                        .expect("`changed_position_idx` must be a valid index in `forces`"),
                )
        }

        fn calculate_group_potential_diff(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
        ) -> T {
            #[allow(deprecated)]
            self.into_iter()
                .nth(changed_position_idx)
                .expect("There must be at least one atom-decoupled potential in the slice")
                .calculate_atom_potential_diff(
                    changed_position_idx,
                    old_value,
                    group,
                    positions
                        .get(changed_position_idx)
                        .expect("`changed_position_idx` must be a valid index in `positions`"),
                )
        }

        fn update_group_forces(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) {
            #[allow(deprecated)]
            self.into_iter()
                .nth(changed_position_idx)
                .expect("There must be at least one atom-decoupled potential in the slice")
                .update_atom_force(
                    changed_position_idx,
                    old_value,
                    group,
                    positions
                        .get(changed_position_idx)
                        .expect("`changed_position_idx` must be a valid index in `positions`"),
                    forces
                        .get_mut(changed_position_idx)
                        .expect("`changed_position_idx` must be a valid index in `forces`"),
                )
        }
    }
}
