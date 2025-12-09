use super::AtomDecoupledPhysicalPotential;
use crate::core::AtomGroupInfo;
use std::ops::Add;

#[cfg(feature = "monte_carlo")]
mod monte_carlo;

/// A trait for physical potentials that yield the contribution of a single group
/// in a given replica to the total potential energy of said replica.
///
/// Any implementor of this trait automatically implements [`PhysicalPotential`]
///
/// [`PhysicalPotential`]: super::PhysicalPotential
pub trait GroupDecoupledPhysicalPotential<T, V> {
    /// Calculates the contribution of this group to the total physical potential energy
    /// of the replica and sets the group_forces of this group accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the replica and adds the group_forces arising from this potential to the group_forces of this group.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the replica.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(&mut self, group: &AtomGroupInfo<T>, group_positions: &[V]) -> T;

    /// Sets the group_forces of this group.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        group_positions: &[V],
        group_forces: &mut [V],
    );

    /// Adds the group_forces arising from this potential to the group_forces of this group.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        group_positions: &[V],
        group_forces: &mut [V],
    );
}

impl<T, V, U> GroupDecoupledPhysicalPotential<T, V> for U
where
    T: Add<Output = T>,
    U: ?Sized + AtomDecoupledPhysicalPotential<T, V>,
{
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        let mut iter = group_positions
            .iter()
            .zip(group_forces.iter_mut())
            .enumerate()
            .map(|(idx, (position, force))| {
                AtomDecoupledPhysicalPotential::calculate_potential_set_force(
                    self, group, idx, position, force,
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
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        let mut iter = group_positions
            .iter()
            .zip(group_forces.iter_mut())
            .enumerate()
            .map(|(idx, (position, force))| {
                AtomDecoupledPhysicalPotential::calculate_potential_add_force(
                    self, group, idx, position, force,
                )
            });
        let first_atom_energy = iter
            .next()
            .expect("There must be at least one atom in a group");
        iter.fold(first_atom_energy, |accum_energy, atom_energy| {
            accum_energy + atom_energy
        })
    }

    fn calculate_potential(&mut self, group: &AtomGroupInfo<T>, group_positions: &[V]) -> T {
        let mut iter = group_positions.iter().enumerate().map(|(idx, position)| {
            #[allow(deprecated)]
            AtomDecoupledPhysicalPotential::calculate_potential(self, group, idx, position)
        });
        let first_atom_energy = iter
            .next()
            .expect("There must be at least one atom in a group");
        iter.fold(first_atom_energy, |accum_energy, atom_energy| {
            accum_energy + atom_energy
        })
    }

    fn set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        group_positions: &[V],
        group_forces: &mut [V],
    ) {
        for (idx, (position, force)) in group_positions
            .iter()
            .zip(group_forces.iter_mut())
            .enumerate()
        {
            #[allow(deprecated)]
            AtomDecoupledPhysicalPotential::set_force(self, group, idx, position, force);
        }
    }

    fn add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        group_positions: &[V],
        group_forces: &mut [V],
    ) {
        for (idx, (position, force)) in group_positions
            .iter()
            .zip(group_forces.iter_mut())
            .enumerate()
        {
            #[allow(deprecated)]
            AtomDecoupledPhysicalPotential::set_force(self, group, idx, position, force);
        }
    }
}

#[cfg(feature = "monte_carlo")]
pub use monte_carlo::MonteCarloGroupDecoupledPhysicalPotential;
