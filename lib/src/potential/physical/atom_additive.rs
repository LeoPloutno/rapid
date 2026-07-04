use super::PhysicalPotential;
use crate::{
    core::{
        MeaningfulOutput,
        error::{EmptyError, InvalidIndexError},
        sync_ops::{SyncAddReceiver, SyncAddSender},
    },
    potential::GroupInTypeInImage,
};
use macros::efficient_alternatives;
use std::{
    ops::{Add, AddAssign},
    sync::{
        Barrier, RwLock,
        mpsc::{Receiver, Sender},
    },
};

#[cfg(feature = "monte_carlo")]
mod monte_carlo;
#[cfg(feature = "monte_carlo")]
pub use monte_carlo::AtomAdditiveMonteCarloPhysicalPotential;

#[doc =
cfg_select! {
    feature = "monte_carlo" => "A wrapper for implementors of the [`AtomAdditivePhysicalPotential`] and [`AtomAdditiveMonteCarloPhysicalPotential`] traits.",
    _ => "A wrapper for implementors of the [`AtomAdditivePhysicalPotential`] trait."
}
]
pub struct AdditivePhysicalPotential<P: ?Sized, D> {
    data: D,
    potential: P,
}

impl<P> AdditivePhysicalPotential<P, ()> {
    /// Wraps the provided value with `AdditivePhysicalPotential`.
    pub const fn new(potential: P) -> Self {
        Self {
            data: (),
            potential,
        }
    }
}

impl<T, P> AdditivePhysicalPotential<P, Sender<T>> {
    /// Wraps the provided value with `AdditivePhysicalPotential`.
    pub const fn new(potential: P, channel: Sender<T>) -> Self {
        Self {
            data: channel,
            potential,
        }
    }
}

impl<T, P> AdditivePhysicalPotential<P, Receiver<T>> {
    /// Wraps the provided value with `AdditivePhysicalPotential`.
    pub const fn new(potential: P, channel: Receiver<T>) -> Self {
        Self {
            data: channel,
            potential,
        }
    }
}

/// A trait for physical potentials that can be expressed as a sum
/// of potentials that each depend only on a single atom.
///
/// For any type `P` that implements this trait, [`AdditivePhysicalPotential<P>`]
/// automatically implements [`PhysicalPotential`].
pub trait AtomAdditivePhysicalPotential<T: Add<Output = T>, V> {
    /// The type of error `Self` returns.
    type AtomError;
    /// The type of error [`AdditivePhysicalPotential<Self>`] returns.
    type SystemError: From<Self::AtomError> + From<EmptyError> + From<InvalidIndexError>;

    /// Calculates the contribution of an atom to the total physical potential energy
    /// of the image and the force arising from it.
    ///
    /// Returns the contribution to the potential energy and the force.
    fn calculate_energy_and_force(
        &mut self,
        atom_index: usize,
        position: &V,
    ) -> Result<(T, V), Self::AtomError>;

    /// Calculates the contribution of an atom to the total physical potential energy
    /// of the image.
    ///
    /// Returns the contribution to the potential energy.
    #[efficient_alternatives("calculate_energy_and_force")]
    fn calculate_energy(&mut self, atom_index: usize, position: &V) -> Result<T, Self::AtomError>;

    /// Calculates the force arising from this potential.
    ///
    /// Returns the force.
    #[efficient_alternatives("calculate_energy_and_force")]
    fn calculate_force(&mut self, atom_index: usize, position: &V) -> Result<V, Self::AtomError>;
}

impl<T, V, P, D> AtomAdditivePhysicalPotential<T, V> for AdditivePhysicalPotential<P, D>
where
    T: Add<Output = T>,
    V: AddAssign,
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

impl<T, V, A, M, P, D> PhysicalPotential<T, V, A, M, ()> for AdditivePhysicalPotential<P, D>
where
    T: Add<Output = T>,
    V: AddAssign,
    A: SyncAddSender<T> + ?Sized,
    M: ?Sized,
    P: ?Sized,
    Self: AtomAdditivePhysicalPotential<T, V>,
    <Self as AtomAdditivePhysicalPotential<T, V>>::SystemError: From<A::Error>,
{
    type Error = <Self as AtomAdditivePhysicalPotential<T, V>>::SystemError;

    fn calculate_energy_set_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        let mut iter = positions
            .read()
            .iter()
            .enumerate()
            .map(|(index, position)| {
                AtomAdditivePhysicalPotential::calculate_energy_and_force(self, index, position)
            });
        let (first_atom_energy, first_atom_force) = iter.next().ok_or(EmptyError)??;
        let (force, forces) = forces.split_first_mut().ok_or(EmptyError)?;
        *force = first_atom_force;
        let group_energy =
            iter.zip(forces)
                .try_fold(first_atom_energy, |accum_energy, (elem, force)| {
                    let (atom_energy, atom_force) = elem?;
                    *force = atom_force;
                    Ok::<_, <Self as AtomAdditivePhysicalPotential<T, V>>::AtomError>(
                        accum_energy + atom_energy,
                    )
                })?;
        adder.send(group_energy)?;
        Ok(())
    }

    fn calculate_energy_add_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        let mut iter = positions
            .read()
            .iter()
            .enumerate()
            .map(|(index, position)| {
                AtomAdditivePhysicalPotential::calculate_energy_and_force(self, index, position)
            });
        let (first_atom_energy, first_atom_force) = iter.next().ok_or(EmptyError)??;
        let (force, forces) = forces.split_first_mut().ok_or(EmptyError)?;
        *force += first_atom_force;
        let group_energy =
            iter.zip(forces)
                .try_fold(first_atom_energy, |accum_energy, (elem, force)| {
                    let (atom_energy, atom_force) = elem?;
                    *force += atom_force;
                    Ok::<_, <Self as AtomAdditivePhysicalPotential<T, V>>::AtomError>(
                        accum_energy + atom_energy,
                    )
                })?;
        adder.send(group_energy)?;
        Ok(())
    }

    fn calculate_energy(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<(), Self::Error> {
        let mut iter = positions
            .read()
            .iter()
            .enumerate()
            .map(|(index, position)| {
                #[allow(deprecated)]
                AtomAdditivePhysicalPotential::calculate_energy(self, index, position)
            });
        let first_atom_energy = iter.next().ok_or(EmptyError)??;
        let group_energy = iter.try_fold(first_atom_energy, |accum_energy, atom_energy| {
            Ok::<_, <Self as AtomAdditivePhysicalPotential<T, V>>::AtomError>(
                accum_energy + atom_energy?,
            )
        })?;
        adder.send(group_energy)?;
        Ok(())
    }

    fn set_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        for (index, (position, force)) in positions.read().iter().zip(forces).enumerate() {
            *force = self.calculate_force(index, position)?;
        }
        Ok(())
    }

    fn add_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        for (index, (position, force)) in positions.read().iter().zip(forces).enumerate() {
            *force += self.calculate_force(index, position)?;
        }
        Ok(())
    }
}

impl<T, V, A, M, P, D> PhysicalPotential<T, V, A, M, T> for AdditivePhysicalPotential<P, D>
where
    T: MeaningfulOutput + Add<Output = T>,
    V: AddAssign,
    A: SyncAddReceiver<T> + ?Sized,
    M: ?Sized,
    P: ?Sized,
    Self: AtomAdditivePhysicalPotential<T, V>,
    <Self as AtomAdditivePhysicalPotential<T, V>>::SystemError: From<A::Error>,
{
    type Error = <Self as AtomAdditivePhysicalPotential<T, V>>::SystemError;

    fn calculate_energy_set_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<T, Self::Error> {
        let mut iter = positions
            .read()
            .iter()
            .enumerate()
            .map(|(index, position)| {
                AtomAdditivePhysicalPotential::calculate_energy_and_force(self, index, position)
            });
        let (first_atom_energy, first_atom_force) = iter.next().ok_or(EmptyError)??;
        let (force, forces) = forces.split_first_mut().ok_or(EmptyError)?;
        *force = first_atom_force;
        let group_energy =
            iter.zip(forces)
                .try_fold(first_atom_energy, |accum_energy, (elem, force)| {
                    let (atom_energy, atom_force) = elem?;
                    *force = atom_force;
                    Ok::<_, <Self as AtomAdditivePhysicalPotential<T, V>>::AtomError>(
                        accum_energy + atom_energy,
                    )
                })?;
        let image_energy = group_energy + adder.recv_sum()?.ok_or(EmptyError)?;
        Ok(image_energy)
    }

    fn calculate_energy_add_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<T, Self::Error> {
        let mut iter = positions
            .read()
            .iter()
            .enumerate()
            .map(|(index, position)| {
                AtomAdditivePhysicalPotential::calculate_energy_and_force(self, index, position)
            });
        let (first_atom_energy, first_atom_force) = iter.next().ok_or(EmptyError)??;
        let (force, forces) = forces.split_first_mut().ok_or(EmptyError)?;
        *force += first_atom_force;
        let group_energy =
            iter.zip(forces)
                .try_fold(first_atom_energy, |accum_energy, (elem, force)| {
                    let (atom_energy, atom_force) = elem?;
                    *force += atom_force;
                    Ok::<_, <Self as AtomAdditivePhysicalPotential<T, V>>::AtomError>(
                        accum_energy + atom_energy,
                    )
                })?;
        let image_energy = group_energy + adder.recv_sum()?.ok_or(EmptyError)?;
        Ok(image_energy)
    }

    fn calculate_energy(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<T, Self::Error> {
        let mut iter = positions
            .read()
            .iter()
            .enumerate()
            .map(|(index, position)| {
                #[allow(deprecated)]
                AtomAdditivePhysicalPotential::calculate_energy(self, index, position)
            });
        let first_atom_energy = iter.next().ok_or(EmptyError)??;
        let group_energy = iter.try_fold(first_atom_energy, |accum_energy, atom_energy| {
            Ok::<_, <Self as AtomAdditivePhysicalPotential<T, V>>::AtomError>(
                accum_energy + atom_energy?,
            )
        })?;
        let image_energy = group_energy + adder.recv_sum()?.ok_or(EmptyError)?;
        Ok(image_energy)
    }

    fn set_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        for (index, (position, force)) in positions.read().iter().zip(forces).enumerate() {
            *force = self.calculate_force(index, position)?;
        }
        Ok(())
    }

    fn add_forces(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        for (index, (position, force)) in positions.read().iter().zip(forces).enumerate() {
            *force += self.calculate_force(index, position)?;
        }
        Ok(())
    }
}
