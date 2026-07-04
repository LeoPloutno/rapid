use super::{
    super::MonteCarloPhysicalPotential, AdditivePhysicalPotential, AtomAdditivePhysicalPotential,
};
use crate::{
    core::{
        MeaningfulOutput,
        error::InvalidIndexError,
        monte_carlo::ChangedGroup,
        sync_ops::{SyncAddReceiver, SyncAddSender},
    },
    potential::GroupInTypeInImage,
};
use macros::efficient_alternatives;
use std::{
    ops::{Add, AddAssign},
    sync::{
        Barrier, RwLock,
        mpsc::{Receiver, RecvError, SendError, Sender},
    },
};

/// A trait for atom-additive physical potentials that may be used in a Monte-Carlo algorithm.
///
/// For any type `P` that implements this trait, [`dditiveMonteCarloPhysicalPotential<P>`]
/// atomatically implements [`MonteCarloPhysicalPotential`].
pub trait AtomAdditiveMonteCarloPhysicalPotential<T, V>:
    AtomAdditivePhysicalPotential<T, V>
where
    T: Add<Output = T>,
{
    /// The type of error `Self` returns.
    type AtomError;
    /// The type of error [`dditiveMonteCarloPhysicalPotential<Self>`] returns.
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

impl<T, V, P, C> AtomAdditiveMonteCarloPhysicalPotential<T, V> for AdditivePhysicalPotential<P, C>
where
    T: Add<Output = T>,
    V: AddAssign,
    P: AtomAdditiveMonteCarloPhysicalPotential<T, V> + ?Sized,
{
    type AtomError = <P as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::AtomError;
    type SystemError = <P as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::SystemError;

    fn calculate_new_energy_and_force(
        &mut self,
        atom_index: usize,
        old_energy: T,
        old_position: V,
        position: &V,
    ) -> Result<(T, V), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::AtomError> {
        self.potential.calculate_new_energy_and_force(
            atom_index,
            old_energy,
            old_position,
            position,
        )
    }

    fn calculate_new_energy(
        &mut self,
        atom_index: usize,
        old_energy: T,
        old_position: V,
        position: &V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::AtomError> {
        #[allow(deprecated)]
        self.potential
            .calculate_new_energy(atom_index, old_energy, old_position, position)
    }
}

impl<T, V, A, M, P> MonteCarloPhysicalPotential<T, V, A, M, ()>
    for AdditivePhysicalPotential<P, Sender<T>>
where
    T: Add<Output = T>,
    V: AddAssign,
    A: SyncAddSender<T> + ?Sized,
    M: ?Sized,
    Self: AtomAdditiveMonteCarloPhysicalPotential<T, V>,
    <Self as AtomAdditivePhysicalPotential<T, V>>::SystemError: From<A::Error>,
    <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::SystemError:
        From<A::Error> + From<SendError<T>>,
{
    type Error = <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::SystemError;

    fn calculate_new_energy_set_changed_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, A, M, ()>>::Error> {
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
            self.data.send(new_energy)?;
        }
        Ok(())
    }

    fn calculate_new_energy_add_changed_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, A, M, ()>>::Error> {
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
            self.data.send(new_energy)?;
        }
        Ok(())
    }

    fn calculate_new_energy(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, A, M, ()>>::Error> {
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
            self.data.send(new_energy)?;
        }
        Ok(())
    }

    fn set_changed_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, A, M, ()>>::Error> {
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
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, A, M, ()>>::Error> {
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

impl<T, V, A, M, P> MonteCarloPhysicalPotential<T, V, A, M, T>
    for AdditivePhysicalPotential<P, Receiver<T>>
where
    T: Add<Output = T> + MeaningfulOutput,
    V: AddAssign,
    A: SyncAddReceiver<T> + ?Sized,
    M: ?Sized,
    Self: AtomAdditiveMonteCarloPhysicalPotential<T, V>,
    <Self as AtomAdditivePhysicalPotential<T, V>>::SystemError: From<A::Error>,
    <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::SystemError:
        From<A::Error> + From<RecvError>,
{
    type Error = <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::SystemError;

    fn calculate_new_energy_set_changed_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, A, M, T>>::Error> {
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
            Ok(self.data.recv()?)
        }
    }

    fn calculate_new_energy_add_changed_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, A, M, T>>::Error> {
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
            Ok(self.data.recv()?)
        }
    }

    fn calculate_new_energy(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, A, M, T>>::Error> {
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
            Ok(self.data.recv()?)
        }
    }

    fn set_changed_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, A, M, T>>::Error> {
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
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, A, M, T>>::Error> {
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
