use super::AtomDecoupledPhysicalPotential;
use crate::core::AtomGroupInfo;
use std::ops::Add;

/// A trait for physical potentials that yield the contribution of a single group
/// in a given replica to the total potential energy of said replica.
///
/// Any implementor of this trait automatically implements [`PhysicalPotential`]
///
/// [`PhysicalPotential`]: super::PhysicalPotential
pub trait GroupDecoupledPhysicalPotential<T, V> {
    /// Calculates the contribution of this group to the total potential energy
    /// of the replica and sets the forces of this group accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the total potential energy
    /// of the replica and adds the forces arising from this potential to the forces of this group.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
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
    fn calculate_potential(&mut self, group: &AtomGroupInfo<T>, positions: &[V]) -> T;

    /// Sets the forces of this group.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(&mut self, group: &AtomGroupInfo<T>, positions: &[V], forces: &mut [V]);

    /// Adds the forces arising from this potential to the forces of this group.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(&mut self, group: &AtomGroupInfo<T>, positions: &[V], forces: &mut [V]);
}

impl<T, V, U> GroupDecoupledPhysicalPotential<T, V> for U
where
    T: Add<Output = T>,
    U: ?Sized + AtomDecoupledPhysicalPotential<T, V>,
{
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        let mut iter =
            positions
                .iter()
                .zip(forces.iter_mut())
                .enumerate()
                .map(|(idx, (position, force))| {
                    AtomDecoupledPhysicalPotential::calculate_potential_set_force(
                        self, idx, group, position, force,
                    )
                });
        let first_atom_energy = iter
            .next()
            .expect("There must be at least one atom in a group");
        iter.fold(first_atom_energy, |accum_energy, atom_energy| {
            accum_energy + atom_energy
        })
    }

    fn calculate_potential_add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        let mut iter =
            positions
                .iter()
                .zip(forces.iter_mut())
                .enumerate()
                .map(|(idx, (position, force))| {
                    AtomDecoupledPhysicalPotential::calculate_potential_add_force(
                        self, idx, group, position, force,
                    )
                });
        let first_atom_energy = iter
            .next()
            .expect("There must be at least one atom in a group");
        iter.fold(first_atom_energy, |accum_energy, atom_energy| {
            accum_energy + atom_energy
        })
    }

    fn calculate_potential(&mut self, group: &AtomGroupInfo<T>, positions: &[V]) -> T {
        let mut iter = positions.iter().enumerate().map(|(idx, position)| {
            #[allow(deprecated)]
            AtomDecoupledPhysicalPotential::calculate_potential(self, idx, group, position)
        });
        let first_atom_energy = iter
            .next()
            .expect("There must be at least one atom in a group");
        iter.fold(first_atom_energy, |accum_energy, atom_energy| {
            accum_energy + atom_energy
        })
    }

    fn set_forces(&mut self, group: &AtomGroupInfo<T>, positions: &[V], forces: &mut [V]) {
        for (idx, (position, force)) in positions.iter().zip(forces.iter_mut()).enumerate() {
            #[allow(deprecated)]
            AtomDecoupledPhysicalPotential::set_force(self, idx, group, position, force);
        }
    }

    fn add_forces(&mut self, group: &AtomGroupInfo<T>, positions: &[V], forces: &mut [V]) {
        for (idx, (position, force)) in positions.iter().zip(forces.iter_mut()).enumerate() {
            #[allow(deprecated)]
            AtomDecoupledPhysicalPotential::set_force(self, idx, group, position, force);
        }
    }
}

#[cfg(feature = "monte_carlo")]
pub(super) mod monte_carlo {
    use std::ops::Add;

    use super::GroupDecoupledPhysicalPotential;
    use crate::{
        core::AtomGroupInfo, potential::physical::MonteCarloAtomDecoupledPhysicalPotential,
    };

    /// A trait for group-deoupled physical potentials that may be used in a Monte-Carlo algorithm.
    ///
    /// Any implementor of this trait automatically implements [`MonteCarloPhysicalPotential`]
    ///
    /// [`MonteCarloPhysicalPotential`]: super::MonteCarloPhysicalPotential
    pub trait MonteCarloGroupDecoupledPhysicalPotential<T, V>:
        GroupDecoupledPhysicalPotential<T, V>
    {
        /// Calculates the change in the potential energy of this group
        /// after a change in the position of one of its atoms
        /// and sets the forces of this group accordingly.
        ///
        /// Returns the change in potential energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff_set_changed_forces(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T;

        /// Calculates the change in the potential energy of this group
        /// after a change in the position of one of its atoms
        /// and adds the updated forces to the forces of this group.
        ///
        /// Returns the change in potential energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff_add_changed_forces(
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
        #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
        ) -> T;

        /// Updates the forces of this group after a change in the position of one of its atoms.
        #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
        fn set_changed_forces(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        );

        /// Adds the updated forces to the forces of this group given a change
        /// in the position of one of its atoms.
        #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
        fn add_changed_forces(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        );
    }

    impl<T, V, U> MonteCarloGroupDecoupledPhysicalPotential<T, V> for U
    where
        T: Add<Output = T>,
        U: ?Sized + MonteCarloAtomDecoupledPhysicalPotential<T, V>,
    {
        fn calculate_potential_diff_set_changed_forces(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T {
            MonteCarloAtomDecoupledPhysicalPotential::calculate_potential_diff_set_changed_force(
                self,
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

        fn calculate_potential_diff_add_changed_forces(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T {
            MonteCarloAtomDecoupledPhysicalPotential::calculate_potential_diff_add_changed_force(
                self,
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

        fn calculate_potential_diff(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
        ) -> T {
            #[allow(deprecated)]
            MonteCarloAtomDecoupledPhysicalPotential::calculate_potential_diff(
                self,
                changed_position_idx,
                old_value,
                group,
                positions
                    .get(changed_position_idx)
                    .expect("`changed_position_idx` must be a valid index in `positions`"),
            )
        }

        fn set_changed_forces(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) {
            #[allow(deprecated)]
            MonteCarloAtomDecoupledPhysicalPotential::set_changed_force(
                self,
                changed_position_idx,
                old_value,
                group,
                positions
                    .get(changed_position_idx)
                    .expect("`changed_position_idx` must be a valid index in `positions`"),
                forces
                    .get_mut(changed_position_idx)
                    .expect("`changed_position_idx` must be a valid index in `forces`"),
            );
        }

        fn add_changed_forces(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) {
            #[allow(deprecated)]
            MonteCarloAtomDecoupledPhysicalPotential::add_changed_force(
                self,
                changed_position_idx,
                old_value,
                group,
                positions
                    .get(changed_position_idx)
                    .expect("`changed_position_idx` must be a valid index in `positions`"),
                forces
                    .get_mut(changed_position_idx)
                    .expect("`changed_position_idx` must be a valid index in `forces`"),
            );
        }
    }
}
