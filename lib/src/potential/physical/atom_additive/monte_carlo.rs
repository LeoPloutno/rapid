use super::{AdditivePhysicalPotential, AtomAdditivePhysicalPotential};
use crate::{
    core::{
        GroupInTypeInImage,
        error::InvalidIndexError,
        marker::MeaningfulOutput,
        sync_ops::{SyncAddReceiver, SyncAddSender},
    },
    potential::physical::{ChangedGroup, MonteCarloPhysicalPotential},
};
use macros::efficient_alternatives;
use std::{
    ops::{Add, AddAssign},
    sync::mpsc::{Receiver, RecvError, SendError, Sender},
};

/// A trait for atom-additive physical potentials that may be used in a Monte-Carlo algorithm.
///
/// For any type `P` that implements this trait, [`AdditivePhysicalPotential<_, MonteCarlo<_, P>>`]
/// atomatically implements [`MonteCarloPhysicalPotential`].
pub trait AtomAdditiveMonteCarloPhysicalPotential<T, V>:
    AtomAdditivePhysicalPotential<T, V>
where
    T: Add<Output = T>,
{
    /// The type of error `Self` returns.
    type AtomError;
    /// The type of error [`AdditiveMonteCarloPhysicalPotential<Self>`] returns.
    type SystemError: From<<Self as AtomAdditivePhysicalPotential<T, V>>::AtomError>
        + From<<Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::AtomError>
        + From<InvalidIndexError>;

    /// Calculates the contribution of an atom to the total physical potential energy
    /// of the image and the force arising from it after a change in the atom's position.
    ///
    /// Returns the contribution to the potential energy and the force.
    fn calculate_new_energy_and_force(
        &mut self,
        atom_index: usize,
        old_energy: T,
        old_position: V,
        position: &V,
    ) -> Result<(T, V), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::AtomError>;

    /// Calculates the contribution of an atom to the total physical potential energy
    /// of the image after a change in the atom's position.
    ///
    /// Returns the contribution to the potential energy.
    #[efficient_alternatives("update_energy_and_force")]
    fn calculate_new_energy(
        &mut self,
        atom_index: usize,
        old_energy: T,
        old_position: V,
        position: &V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::AtomError>;
}

/// A wrapper for implementors of the [`AtomAdditiveMonteCarloPhysicalPotential`] trait.
pub struct MonteCarlo<C, P: ?Sized> {
    channel: C,
    potential: P,
}

impl<T, V, C, P> AtomAdditivePhysicalPotential<T, V> for MonteCarlo<C, P>
where
    T: Add<Output = T>,
    P: AtomAdditivePhysicalPotential<T, V> + ?Sized,
{
    type AtomError = P::AtomError;
    type SystemError = P::SystemError;

    #[inline(always)]
    fn calculate_energy_and_force(
        &mut self,
        atom_index: usize,
        position: &V,
    ) -> Result<(T, V), Self::AtomError> {
        #[allow(deprecated)]
        self.potential
            .calculate_energy_and_force(atom_index, position)
    }

    #[inline(always)]
    fn calculate_energy(&mut self, atom_index: usize, position: &V) -> Result<T, Self::AtomError> {
        #[allow(deprecated)]
        self.potential.calculate_energy(atom_index, position)
    }

    #[inline(always)]
    fn calculate_force(&mut self, atom_index: usize, position: &V) -> Result<V, Self::AtomError> {
        #[allow(deprecated)]
        self.potential.calculate_force(atom_index, position)
    }
}

impl<T, V, A, C, P> AtomAdditiveMonteCarloPhysicalPotential<T, V>
    for AdditivePhysicalPotential<A, MonteCarlo<C, P>>
where
    T: Add<Output = T>,
    V: AddAssign,
    P: AtomAdditiveMonteCarloPhysicalPotential<T, V> + ?Sized,
{
    type AtomError = <P as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::AtomError;
    type SystemError = <P as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::SystemError;

    #[inline(always)]
    fn calculate_new_energy_and_force(
        &mut self,
        atom_index: usize,
        old_energy: T,
        old_position: V,
        position: &V,
    ) -> Result<(T, V), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::AtomError> {
        self.potential.potential.calculate_new_energy_and_force(
            atom_index,
            old_energy,
            old_position,
            position,
        )
    }

    #[inline(always)]
    fn calculate_new_energy(
        &mut self,
        atom_index: usize,
        old_energy: T,
        old_position: V,
        position: &V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::AtomError> {
        #[allow(deprecated)]
        self.potential.potential.calculate_new_energy(
            atom_index,
            old_energy,
            old_position,
            position,
        )
    }
}

impl<T, V, A, P> MonteCarloPhysicalPotential<T, V, ()>
    for AdditivePhysicalPotential<A, MonteCarlo<Sender<T>, P>>
where
    T: Add<Output = T>,
    V: AddAssign,
    A: SyncAddSender<T>,
    P: ?Sized,
    Self: AtomAdditivePhysicalPotential<T, V, SystemError: From<A::Error>>
        + AtomAdditiveMonteCarloPhysicalPotential<
            T,
            V,
            SystemError: From<A::Error> + From<SendError<T>>,
        >,
{
    type Error = <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::SystemError;

    fn calculate_new_energy_set_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, ()>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            let (new_energy, new_force) =
                AtomAdditiveMonteCarloPhysicalPotential::calculate_new_energy_and_force(
                    self,
                    changed_atom_index,
                    old_energy,
                    old_position,
                    positions.get(changed_atom_index).ok_or_else(|| {
                        InvalidIndexError::new(changed_atom_index, positions.len())
                    })?,
                )?;
            let forces_len = forces.len();
            *forces
                .get_mut(changed_atom_index)
                .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_len))? = new_force;
            self.potential.channel.send(new_energy)?;
        }
        Ok(())
    }

    fn calculate_new_energy_add_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, ()>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            let (new_energy, new_force) =
                AtomAdditiveMonteCarloPhysicalPotential::calculate_new_energy_and_force(
                    self,
                    changed_atom_index,
                    old_energy,
                    old_position,
                    positions.get(changed_atom_index).ok_or_else(|| {
                        InvalidIndexError::new(changed_atom_index, positions.len())
                    })?,
                )?;
            let forces_len = forces.len();
            *forces
                .get_mut(changed_atom_index)
                .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_len))? +=
                new_force;
            self.potential.channel.send(new_energy)?;
        }
        Ok(())
    }

    fn calculate_new_energy(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, ()>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            #[allow(deprecated)]
            let new_energy = AtomAdditiveMonteCarloPhysicalPotential::calculate_new_energy(
                self,
                changed_atom_index,
                old_energy,
                old_position,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
            )?;
            self.potential.channel.send(new_energy)?;
        }
        Ok(())
    }

    fn set_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, ()>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            #[allow(deprecated)]
            let new_force = AtomAdditivePhysicalPotential::calculate_force(
                self,
                changed_atom_index,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
            )?;
            let forces_len = forces.len();
            *forces
                .get_mut(changed_atom_index)
                .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_len))? = new_force;
        }
        Ok(())
    }

    fn add_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, ()>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            #[allow(deprecated)]
            let new_force = AtomAdditivePhysicalPotential::calculate_force(
                self,
                changed_atom_index,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
            )?;
            let forces_len = forces.len();
            *forces
                .get_mut(changed_atom_index)
                .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_len))? +=
                new_force;
        }
        Ok(())
    }
}

impl<T, V, A, P> MonteCarloPhysicalPotential<T, V, T>
    for AdditivePhysicalPotential<A, MonteCarlo<Receiver<T>, P>>
where
    T: Add<Output = T> + MeaningfulOutput,
    V: AddAssign,
    A: SyncAddReceiver<T>,
    P: ?Sized,
    Self: AtomAdditivePhysicalPotential<T, V, SystemError: From<A::Error>>
        + AtomAdditiveMonteCarloPhysicalPotential<T, V, SystemError: From<A::Error> + From<RecvError>>,
{
    type Error = <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::SystemError;

    fn calculate_new_energy_set_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, T>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            let (new_energy, new_force) =
                AtomAdditiveMonteCarloPhysicalPotential::calculate_new_energy_and_force(
                    self,
                    changed_atom_index,
                    old_energy,
                    old_position,
                    positions.get(changed_atom_index).ok_or_else(|| {
                        InvalidIndexError::new(changed_atom_index, positions.len())
                    })?,
                )?;
            let forces_len = forces.len();
            *forces
                .get_mut(changed_atom_index)
                .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_len))? = new_force;
            Ok(new_energy)
        } else {
            Ok(self.potential.channel.recv()?)
        }
    }

    fn calculate_new_energy_add_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, T>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            let (new_energy, new_force) =
                AtomAdditiveMonteCarloPhysicalPotential::calculate_new_energy_and_force(
                    self,
                    changed_atom_index,
                    old_energy,
                    old_position,
                    positions.get(changed_atom_index).ok_or_else(|| {
                        InvalidIndexError::new(changed_atom_index, positions.len())
                    })?,
                )?;
            let forces_len = forces.len();
            *forces
                .get_mut(changed_atom_index)
                .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_len))? +=
                new_force;
            Ok(new_energy)
        } else {
            Ok(self.potential.channel.recv()?)
        }
    }

    fn calculate_new_energy(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, T>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            #[allow(deprecated)]
            let new_energy = AtomAdditiveMonteCarloPhysicalPotential::calculate_new_energy(
                self,
                changed_atom_index,
                old_energy,
                old_position,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
            )?;
            Ok(new_energy)
        } else {
            Ok(self.potential.channel.recv()?)
        }
    }

    fn set_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, T>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            #[allow(deprecated)]
            let new_force = AtomAdditivePhysicalPotential::calculate_force(
                self,
                changed_atom_index,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
            )?;
            let forces_len = forces.len();
            *forces
                .get_mut(changed_atom_index)
                .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_len))? = new_force;
        }
        Ok(())
    }

    fn add_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, T>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            #[allow(deprecated)]
            let new_force = AtomAdditivePhysicalPotential::calculate_force(
                self,
                changed_atom_index,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
            )?;
            let forces_len = forces.len();
            *forces
                .get_mut(changed_atom_index)
                .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_len))? +=
                new_force;
        }
        Ok(())
    }
}
