use super::PhysicalPotential;
use crate::{
    ImageHandle, core::error::InvalidIndexError, potential::physical::MonteCarloAtomDecoupledPhysicalPotential,
};

/// A trait for physical potentials that may be used in a Monte-Carlo algorithm.
pub trait MonteCarloPhysicalPotential<T, V>: PhysicalPotential<T, V> {
    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and updates the group_forces of this group accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and adds the updated group_forces to the group_forces of this group.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom.
    ///
    /// Returns the contribution to the change in total energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        changed_group_inedx: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
    ) -> Result<T, Self::Error>;

    /// Updates the group_forces of this group after a change in the position of a single atom.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    fn set_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Adds the updated group_forces to the group_forces of this group given a change
    /// in the position of a single atom.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    fn add_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;
}

impl<T, V, E, U> MonteCarloPhysicalPotential<T, V> for U
where
    T: Default,
    E: From<InvalidIndexError>,
    U: MonteCarloAtomDecoupledPhysicalPotential<T, V, Error = E> + PhysicalPotential<T, V, Error = E> + ?Sized,
{
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        if changed_group_index == groups_positions.index() {
            let group_positions = groups_positions.read().read();
            let group_forces_len = group_forces.len();
            MonteCarloAtomDecoupledPhysicalPotential::calculate_potential_diff_set_changed_force(
                self,
                changed_atom_index,
                old_value,
                group_positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
                group_forces
                    .get_mut(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, group_forces_len))?,
            )
        } else {
            Ok(T::default())
        }
    }

    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        if changed_group_index == groups_positions.index() {
            let group_positions = groups_positions.read().read();
            let group_forces_len = group_forces.len();
            MonteCarloAtomDecoupledPhysicalPotential::calculate_potential_diff_add_changed_force(
                self,
                changed_atom_index,
                old_value,
                group_positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
                group_forces
                    .get_mut(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, group_forces_len))?,
            )
        } else {
            Ok(T::default())
        }
    }

    fn calculate_potential_diff(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
    ) -> Result<T, Self::Error> {
        if changed_group_index == groups_positions.index() {
            let group_positions = groups_positions.read().read();
            #[allow(deprecated)]
            MonteCarloAtomDecoupledPhysicalPotential::calculate_potential_diff(
                self,
                changed_atom_index,
                old_value,
                group_positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
            )
        } else {
            Ok(T::default())
        }
    }

    fn set_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<(), Self::Error> {
        if changed_group_index == groups_positions.index() {
            let group_positions = groups_positions.read().read();
            let group_forces_len = group_forces.len();
            #[allow(deprecated)]
            MonteCarloAtomDecoupledPhysicalPotential::set_changed_force(
                self,
                changed_atom_index,
                old_value,
                group_positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
                group_forces
                    .get_mut(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, group_forces_len))?,
            )
        } else {
            Ok(())
        }
    }

    fn add_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<(), Self::Error> {
        if changed_group_index == groups_positions.index() {
            let group_positions = groups_positions.read().read();
            let group_forces_len = group_forces.len();
            #[allow(deprecated)]
            MonteCarloAtomDecoupledPhysicalPotential::add_changed_force(
                self,
                changed_atom_index,
                old_value,
                group_positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
                group_forces
                    .get_mut(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, group_forces_len))?,
            )
        } else {
            Ok(())
        }
    }
}
