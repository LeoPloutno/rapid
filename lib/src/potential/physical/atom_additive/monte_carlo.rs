use super::AtomAdditivePhysicalPotential;
use crate::{
    core::{Additive as AdditiveMonteCarloPhysicalPotential, GroupTypeHandle, error::InvalidIndexError},
    potential::physical::MonteCarloPhysicalPotential,
};
use macros::efficient_alternatives;
use std::ops::Add;

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

    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its position and sets the force of this atom accordingly.
    ///
    /// Returns the change in physical physical potential energy.
    fn calculate_potential_diff_set_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Calculates the change in the physical potential energy of this atom
    /// after a change in its position and adds the force arising from this potential
    /// to the force of this atom.
    ///
    /// Returns the change in physical potential energy.
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
    /// Returns the change in physical potential energy.
    #[efficient_alternatives(
        "calculate_potential_diff_set_changed_force",
        "calculate_potential_diff_add_changed_force"
    )]
    fn calculate_potential_diff(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Sets the force of this atom after a change in its position.
    #[efficient_alternatives("calculate_potential_diff_set_changed_force")]
    fn set_changed_force(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
        force: &mut V,
    ) -> Result<(), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Adds the force arising from this potential to the force of this atom
    /// after a change in its position.
    #[efficient_alternatives("calculate_potential_diff_add_changed_force")]
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

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
    fn calculate_potential_diff(
        &mut self,
        atom_index: usize,
        old_value: V,
        position: &V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        #[allow(deprecated)]
        self.0.calculate_potential_diff(atom_index, old_value, position)
    }

    #[inline(always)]
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

    #[inline(always)]
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
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions
                .get(changed_group_index)
                .ok_or_else(|| InvalidIndexError::new(changed_group_index, groups_positions.len()))?
                .read();
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
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions
                .get(changed_group_index)
                .ok_or_else(|| InvalidIndexError::new(changed_group_index, groups_positions.len()))?
                .read();
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
        groups_positions: &[GroupTypeHandle<V>],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions
                .get(changed_group_index)
                .ok_or_else(|| InvalidIndexError::new(changed_group_index, groups_positions.len()))?
                .read();
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
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions
                .get(changed_group_index)
                .ok_or_else(|| InvalidIndexError::new(changed_group_index, groups_positions.len()))?
                .read();
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
        groups_positions: &[GroupTypeHandle<V>],
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V>>::Error> {
        if self.group_index() == changed_group_index {
            let group_positions = groups_positions
                .get(changed_group_index)
                .ok_or_else(|| InvalidIndexError::new(changed_group_index, groups_positions.len()))?
                .read();
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
