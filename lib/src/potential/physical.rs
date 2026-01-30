use crate::core::AtomGroupInfo;

mod atom_decoupled;
mod group_decoupled;
#[cfg(feature = "monte_carlo")]
mod monte_carlo;

/// A trait for physical potentials that yield the contribution of a single group
/// in a given replica to the total physical potential energy of said replica.
pub trait PhysicalPotential<T, V> {
    /// Calculates the contribution of this group to the total potential energy
    /// of the replica and sets the forces of this group accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        group_idx: usize,
        groups: &[AtomGroupInfo<T>],
        groups_positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the total potential energy
    /// of the replica and adds the forces arising from this potential to the forces of this group.
    ///
    /// Returns the contribution to the total energy.
    fn calculate_potential_add_forces(
        &mut self,
        group_idx: usize,
        groups: &[AtomGroupInfo<T>],
        groups_positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group to the total potential energy
    /// of the replica.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(&mut self, group_idx: usize, groups: &[AtomGroupInfo<T>], groups_positions: &[V]) -> T;

    /// Sets the forces of this group.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(&mut self, group_idx: usize, groups: &[AtomGroupInfo<T>], groups_positions: &[V], forces: &mut [V]);

    /// Adds the forces arising from this potential to the forces of this group.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(&mut self, group_idx: usize, groups: &[AtomGroupInfo<T>], groups_positions: &[V], forces: &mut [V]);
}

impl<T, V, U> PhysicalPotential<T, V> for U
where
    U: GroupDecoupledPhysicalPotential<T, V> + ?Sized,
{
    fn calculate_potential_set_forces(
        &mut self,
        group_idx: usize,
        groups: &[AtomGroupInfo<T>],
        groups_positions: &[V],
        forces: &mut [V],
    ) -> T {
        let group = groups
            .get(group_idx)
            .expect("`group_idx` should be a valid index in `groups`");
        let positions = groups_positions
            .get(group.span)
            .expect("`span` should be a valid range in `groups_positions`");
        GroupDecoupledPhysicalPotential::calculate_potential_set_forces(self, group, positions, forces)
    }

    fn calculate_potential_add_forces(
        &mut self,
        group_idx: usize,
        groups: &[AtomGroupInfo<T>],
        groups_positions: &[V],
        forces: &mut [V],
    ) -> T {
        let group = groups
            .get(group_idx)
            .expect("`group_idx` should be a valid index in `groups`");
        let positions = groups_positions
            .get(group.span)
            .expect("`span` should be a valid range in `groups_positions`");
        GroupDecoupledPhysicalPotential::calculate_potential_add_forces(self, group, positions, forces)
    }

    fn calculate_potential(&mut self, group_idx: usize, groups: &[AtomGroupInfo<T>], groups_positions: &[V]) -> T {
        let group = groups
            .get(group_idx)
            .expect("`group_idx` should be a valid index in `groups`");
        let positions = groups_positions
            .get(group.span)
            .expect("`span` should be a valid range in `groups_positions`");
        #[allow(deprecated)]
        GroupDecoupledPhysicalPotential::calculate_potential(self, group, positions)
    }

    fn set_forces(&mut self, group_idx: usize, groups: &[AtomGroupInfo<T>], groups_positions: &[V], forces: &mut [V]) {
        let group = groups
            .get(group_idx)
            .expect("`group_idx` should be a valid index in `groups`");
        let positions = groups_positions
            .get(group.span)
            .expect("`span` should be a valid range in `groups_positions`");
        #[allow(deprecated)]
        GroupDecoupledPhysicalPotential::set_forces(self, group, positions, forces);
    }

    fn add_forces(&mut self, group_idx: usize, groups: &[AtomGroupInfo<T>], groups_positions: &[V], forces: &mut [V]) {
        let group = groups
            .get(group_idx)
            .expect("`group_idx` should be a valid index in `groups`");
        let positions = groups_positions
            .get(group.span)
            .expect("`span` should be a valid range in `groups_positions`");
        #[allow(deprecated)]
        GroupDecoupledPhysicalPotential::add_forces(self, group, positions, forces);
    }
}

pub use self::{atom_decoupled::AtomDecoupledPhysicalPotential, group_decoupled::GroupDecoupledPhysicalPotential};
#[cfg(feature = "monte_carlo")]
pub use self::{
    atom_decoupled::MonteCarloAtomDecoupledPhysicalPotential,
    group_decoupled::MonteCarloGroupDecoupledPhysicalPotential, monte_carlo::MonteCarloPhysicalPotential,
};
