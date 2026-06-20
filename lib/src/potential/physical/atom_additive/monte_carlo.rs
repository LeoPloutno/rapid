use super::{AdditivePhysicalPotential, AtomAdditivePhysicalPotential};
use crate::{
    core::{
        error::InvalidIndexError,
        monte_carlo::ChangedGroup,
        sync_ops::{SyncAddReceiver, SyncAddSender, SyncMulReceiver, SyncMulSender},
    },
    potential::{GroupInTypeInImage, physical::MonteCarloPhysicalPotential},
};
use macros::efficient_alternatives;
use std::{
    ops::Add,
    sync::{
        Barrier, RwLock,
        mpsc::{Receiver, RecvError, SendError, Sender},
    },
};

/// A trait for atom-additive physical potentials that may be used in a Monte-Carlo algorithm.
///
/// For any type `P` that implements this trait, [`AdditiveMonteCarloPhysicalPotential<P>`]
/// atomatically implements [`MonteCarloPhysicalPotential`].
pub trait AtomAdditiveMonteCarloPhysicalPotential<T, V>:
    AtomAdditivePhysicalPotential<T, V>
where
    T: Add<Output = T>,
{
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveMonteCarloPhysicalPotential<Self>`] returns.
    type ErrorSystem: From<<Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>
        + From<InvalidIndexError>
        + From<SendError<T>>
        + From<RecvError>;

    /// Calculates the change (`new - old`) in the physical potential energy of this atom
    /// after a change in its position and sets the force of this atom accordingly.
    ///
    /// Returns the change in physical physical potential energy.
    fn calculate_energy_diff_set_changed_force(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
        force: &mut V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Calculates the change (`new - old`) in the physical potential energy of this atom
    /// after a change in its position and adds the force arising from this potential
    /// to the force of this atom.
    ///
    /// Returns the change in physical potential energy.
    fn calculate_energy_diff_add_changed_force(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
        force: &mut V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Calculates the change (`new - old`) in the physical potential energy of this atom
    /// after a change in its position.
    ///
    /// Returns the change in physical potential energy.
    #[efficient_alternatives(
        "calculate_energy_diff_set_changed_force",
        "calculate_energy_diff_add_changed_force"
    )]
    fn calculate_energy_diff(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Sets the force of this atom after a change in its position.
    #[efficient_alternatives("calculate_energy_diff_set_changed_force")]
    fn set_changed_force(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
        force: &mut V,
    ) -> Result<(), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;

    /// Adds the force arising from this potential to the force of this atom
    /// after a change in its position.
    #[efficient_alternatives("calculate_energy_diff_add_changed_force")]
    fn add_changed_force(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
        force: &mut V,
    ) -> Result<(), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom>;
}

impl<T, V, P> AtomAdditiveMonteCarloPhysicalPotential<T, V> for AdditivePhysicalPotential<P>
where
    T: Default + Add<Output = T>,
    P: AtomAdditiveMonteCarloPhysicalPotential<T, V> + ?Sized,
{
    type ErrorAtom = <P as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom;
    type ErrorSystem = <P as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorSystem;

    #[inline(always)]
    fn calculate_energy_diff_set_changed_force(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
        force: &mut V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        self.0
            .calculate_energy_diff_set_changed_force(atom_index, old_position, position, force)
    }

    #[inline(always)]
    fn calculate_energy_diff_add_changed_force(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
        force: &mut V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        self.0
            .calculate_energy_diff_add_changed_force(atom_index, old_position, position, force)
    }

    #[inline(always)]
    fn calculate_energy_diff(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
    ) -> Result<T, <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        #[allow(deprecated)]
        self.0
            .calculate_energy_diff(atom_index, old_position, position)
    }

    #[inline(always)]
    fn set_changed_force(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
        force: &mut V,
    ) -> Result<(), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        #[allow(deprecated)]
        self.0
            .set_changed_force(atom_index, old_position, position, force)
    }

    #[inline(always)]
    fn add_changed_force(
        &mut self,
        atom_index: usize,
        old_position: V,
        position: &V,
        force: &mut V,
    ) -> Result<(), <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorAtom> {
        #[allow(deprecated)]
        self.0
            .add_changed_force(atom_index, old_position, position, force)
    }
}

impl<T, V, AS, MS, AR, MR, P> MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>
    for AdditivePhysicalPotential<P>
where
    T: Add<Output = T>,
    AS: SyncAddSender<T> + ?Sized,
    MS: SyncMulSender<T> + ?Sized,
    AR: SyncAddReceiver<T> + ?Sized,
    MR: SyncMulReceiver<T> + ?Sized,
    P: ?Sized,
    <Self as AtomAdditivePhysicalPotential<T, V>>::ErrorSystem: From<AS::Error> + From<AR::Error>,
    Self: AtomAdditiveMonteCarloPhysicalPotential<T, V>,
    <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorSystem: From<AS::Error>,
{
    type Error = <Self as AtomAdditiveMonteCarloPhysicalPotential<T, V>>::ErrorSystem;

    fn update_energy_set_changed_forces_with_senders(
        &mut self,
        channel: &Sender<T>,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut AS,
        _multiplier: &mut MS,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error> {
        if let ChangedGroup::This = changed_group {
            let forces_group_len = forces_group.len();
            let positions = positions.read();
            let potential_energy_diff =
                AtomAdditiveMonteCarloPhysicalPotential::calculate_energy_diff_set_changed_force(
                    self,
                    changed_atom_index,
                    old_position,
                    positions.get(changed_atom_index).ok_or_else(|| {
                        InvalidIndexError::new(changed_atom_index, positions.len())
                    })?,
                    forces_group.get_mut(changed_atom_index).ok_or_else(|| {
                        InvalidIndexError::new(changed_atom_index, forces_group_len)
                    })?,
                )?;
            channel.send(potential_energy_diff)?;
        }
        Ok(())
    }

    fn update_energy_set_changed_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut AR,
        _multiplier: &mut MR,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error> {
        Ok(if let ChangedGroup::This = changed_group {
            let forces_group_len = forces_group.len();
            let positions = positions.read();
            AtomAdditiveMonteCarloPhysicalPotential::calculate_energy_diff_set_changed_force(
                self,
                changed_atom_index,
                old_position,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
                forces_group
                    .get_mut(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_group_len))?,
            )?
        } else {
            channel.recv()?
        } + old_energy)
    }

    fn update_energy_add_changed_forces_with_senders(
        &mut self,
        channel: &Sender<T>,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut AS,
        _multiplier: &mut MS,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error> {
        if let ChangedGroup::This = changed_group {
            let forces_group_len = forces_group.len();
            let positions = positions.read();
            let potential_energy_diff =
                AtomAdditiveMonteCarloPhysicalPotential::calculate_energy_diff_add_changed_force(
                    self,
                    changed_atom_index,
                    old_position,
                    positions.get(changed_atom_index).ok_or_else(|| {
                        InvalidIndexError::new(changed_atom_index, positions.len())
                    })?,
                    forces_group.get_mut(changed_atom_index).ok_or_else(|| {
                        InvalidIndexError::new(changed_atom_index, forces_group_len)
                    })?,
                )?;
            channel.send(potential_energy_diff)?;
        }
        Ok(())
    }

    fn update_energy_add_changed_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut AR,
        _multiplier: &mut MR,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error> {
        Ok(if let ChangedGroup::This = changed_group {
            let forces_group_len = forces_group.len();
            let positions = positions.read();
            AtomAdditiveMonteCarloPhysicalPotential::calculate_energy_diff_add_changed_force(
                self,
                changed_atom_index,
                old_position,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
                forces_group
                    .get_mut(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_group_len))?,
            )?
        } else {
            channel.recv()?
        } + old_energy)
    }

    fn update_energy_with_senders(
        &mut self,
        channel: &Sender<T>,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut AS,
        _multiplier: &mut MS,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        _old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error> {
        if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            #[allow(deprecated)]
            let potential_energy_diff =
                AtomAdditiveMonteCarloPhysicalPotential::calculate_energy_diff(
                    self,
                    changed_atom_index,
                    old_position,
                    positions.get(changed_atom_index).ok_or_else(|| {
                        InvalidIndexError::new(changed_atom_index, positions.len())
                    })?,
                )?;
            channel.send(potential_energy_diff)?;
        }
        Ok(())
    }

    fn update_energy_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut AR,
        _multiplier: &mut MR,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error> {
        Ok(if let ChangedGroup::This = changed_group {
            let positions = positions.read();
            #[allow(deprecated)]
            AtomAdditiveMonteCarloPhysicalPotential::calculate_energy_diff(
                self,
                changed_atom_index,
                old_position,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
            )?
        } else {
            channel.recv()?
        } + old_energy)
    }

    fn set_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error> {
        if let ChangedGroup::This = changed_group {
            let forces_group_len = forces_group.len();
            let positions = positions.read();
            #[allow(deprecated)]
            AtomAdditiveMonteCarloPhysicalPotential::set_changed_force(
                self,
                changed_atom_index,
                old_position,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
                forces_group
                    .get_mut(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_group_len))?,
            )?;
        }
        Ok(())
    }

    fn add_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error> {
        if let ChangedGroup::This = changed_group {
            let forces_group_len = forces_group.len();
            let positions = positions.read();
            #[allow(deprecated)]
            AtomAdditiveMonteCarloPhysicalPotential::add_changed_force(
                self,
                changed_atom_index,
                old_position,
                positions
                    .get(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, positions.len()))?,
                forces_group
                    .get_mut(changed_atom_index)
                    .ok_or_else(|| InvalidIndexError::new(changed_atom_index, forces_group_len))?,
            )?;
        }
        Ok(())
    }
}
