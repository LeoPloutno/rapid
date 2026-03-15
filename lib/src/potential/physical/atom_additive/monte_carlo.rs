use std::ops::Add;

use crate::{
    ImageHandle,
    core::{Additive as AdditiveMonteCarloPhysicalPotential, error::InvalidIndexError},
    potential::physical::MonteCarloPhysicalPotential,
};

use super::AtomAdditivePhysicalPotential;

/// A trait for atom-additive physical potentials that may be used in a Monte-Carlo algorithm.
///
/// For any type `P` that implements this trait, [`AdditiveMonteCarloPhysicalPotential<P>`]
/// atomatically implements [`MonteCarloPhysicalPotential`].
pub trait AtomAdditiveMonteCarloPhysicalPotential<T, V>: AtomAdditivePhysicalPotential<T, V>
where
    T: Default + Add<Output = T>,
{
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveMonteCarloPhysicalPotential<Self>`] returns.
    type ErrorSystem: From<<Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> + From<InvalidIndexError>;

    /// Returns the index of the group allocated to this potential amongst all groups of all types.
    fn group_index(&self) -> usize;

    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its position and updates the force of this atom accordingly.
    ///
    /// Returns the change in potential energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its position and adds the updated force to the force
    /// of this atom.
    ///
    /// Returns the change in potential energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its position.
    ///
    /// Returns the change in potential energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_force` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Updates the force of this atom after a change to its position.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_force` as a more efficient alternative"]
    fn set_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<(), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Adds the updated force to the force of this atom given a change in its position.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_force` as a more efficient alternative"]
    fn add_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<(), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;
}

impl<T, V, P> AtomAdditiveMonteCarloPhysicalPotential<T, V> for AdditiveMonteCarloPhysicalPotential<P>
where
    T: Default + Add<Output = T>,
    P: AtomAdditiveMonteCarloPhysicalPotential<T, V> + ?Sized,
{
    type ErrorAtom = <P as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom;
    type ErrorSystem = <P as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorSystem;

    fn group_index(&self) -> usize {
        self.0.group_index()
    }

    fn calculate_potential_diff_set_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        self.0
            .calculate_potential_diff_set_changed_force(atom_index, old_value, position, force)
    }

    fn calculate_potential_diff_add_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        self.0
            .calculate_potential_diff_add_changed_force(atom_index, old_value, position, force)
    }

    fn calculate_potential_diff(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        #[allow(deprecated)]
        self.0.calculate_potential_diff(atom_index, old_value, position)
    }

    fn set_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<(), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        #[allow(deprecated)]
        self.0.set_changed_force(atom_index, old_value, position, force)
    }

    fn add_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<(), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        #[allow(deprecated)]
        self.0.add_changed_force(atom_index, old_value, position, force)
    }
}

impl<T, V, P> MonteCarloPhysicalPotential<T, V> for AdditiveMonteCarloPhysicalPotential<P>
where
    T: Default + Add<Output = T>,
    P: ?Sized,
    Self: AtomAdditiveMonteCarloPhysicalPotential<T, V>,
{
    type Error = <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorSystem;

    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions.read().read();
            let group_forces_len = group_forces.len();
            Ok(
                AtomAdditiveMonteCarloPhysicalPotential::calculate_potential_diff_set_changed_force(
                    self,
                    changed_atom_index,
                    old_value,
                    group_positions
                        .get(changed_atom_index)
                        .ok_or(InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
                    group_forces
                        .get_mut(changed_atom_index)
                        .ok_or(InvalidIndexError::new(changed_atom_index, group_forces_len))?,
                )?,
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
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions.read().read();
            let group_forces_len = group_forces.len();
            Ok(
                AtomAdditiveMonteCarloPhysicalPotential::calculate_potential_diff_add_changed_force(
                    self,
                    changed_atom_index,
                    old_value,
                    group_positions
                        .get(changed_atom_index)
                        .ok_or(InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
                    group_forces
                        .get_mut(changed_atom_index)
                        .ok_or(InvalidIndexError::new(changed_atom_index, group_forces_len))?,
                )?,
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
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions.read().read();
            Ok(
                #[allow(deprecated)]
                AtomAdditiveMonteCarloPhysicalPotential::calculate_potential_diff(
                    self,
                    changed_atom_index,
                    old_value,
                    group_positions
                        .get(changed_atom_index)
                        .ok_or(InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
                )?,
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
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions.read().read();
            let group_forces_len = group_forces.len();
            #[allow(deprecated)]
            AtomAdditiveMonteCarloPhysicalPotential::set_changed_force(
                self,
                changed_atom_index,
                old_value,
                group_positions
                    .get(changed_atom_index)
                    .ok_or(InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
                group_forces
                    .get_mut(changed_atom_index)
                    .ok_or(InvalidIndexError::new(changed_atom_index, group_forces_len))?,
            )?;
        }
        Ok(())
    }

    fn add_changed_forces(
        &mut self,
        changed_group_index: usize,
        changed_atom_index: usize,
        old_value: V,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions.read().read();
            let group_forces_len = group_forces.len();
            #[allow(deprecated)]
            AtomAdditiveMonteCarloPhysicalPotential::add_changed_force(
                self,
                changed_atom_index,
                old_value,
                group_positions
                    .get(changed_atom_index)
                    .ok_or(InvalidIndexError::new(changed_atom_index, group_positions.len()))?,
                group_forces
                    .get_mut(changed_atom_index)
                    .ok_or(InvalidIndexError::new(changed_atom_index, group_forces_len))?,
            )?;
        }
        Ok(())
    }
}
