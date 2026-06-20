use std::sync::{
    Barrier, RwLock,
    mpsc::{Receiver, Sender},
};

use super::PhysicalPotential;
use crate::{
    core::{
        monte_carlo::ChangedGroup,
        sync_ops::{SyncAddReceiver, SyncAddSender, SyncMulReceiver, SyncMulSender},
    },
    potential::GroupInTypeInImage,
};
use macros::{efficient_alternatives, heavy_computation};

/// A trait for physical potentials that may be used in a Monte-Carlo algorithm.
pub trait MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>:
    PhysicalPotential<T, V, AS, MS, AR, MR>
where
    AS: SyncAddSender<T> + ?Sized,
    MS: SyncMulSender<T> + ?Sized,
    AR: SyncAddReceiver<T> + ?Sized,
    MR: SyncMulReceiver<T> + ?Sized,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and sets the forces of this group accordingly.
    ///
    /// Sends the results of the calculations to a receiver.
    #[heavy_computation]
    fn update_energy_set_changed_forces_with_senders(
        &mut self,
        channel: &Sender<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AS,
        multiplier: &mut MS,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and sets the forces of this group accordingly.
    ///
    /// Receives the calculations of other senders and returns the
    /// updated total physical potential energy.
    #[heavy_computation]
    fn update_energy_set_changed_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and adds the forces arising from this potential to the forces of this group.
    ///
    /// Sends the results of the calculations to a receiver.
    #[heavy_computation]
    fn update_energy_add_changed_forces_with_senders(
        &mut self,
        channel: &Sender<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AS,
        multiplier: &mut MS,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom
    /// and adds the forces arising from this potential to the forces of this group.
    ///
    /// Receives the calculations of other senders and returns the
    /// updated total physical potential energy.
    #[heavy_computation]
    fn update_energy_add_changed_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom.
    ///
    /// Sends the results of the calculations to a receiver.
    #[heavy_computation]
    #[efficient_alternatives(
        "update_energy_set_changed_forces_with_senders",
        "update_energy_add_changed_forces_with_senders"
    )]
    fn update_energy_with_senders(
        &mut self,
        channel: &Sender<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AS,
        multiplier: &mut MS,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group to the change in total physical
    /// potential energy of the image after a change in the position of a single atom.
    ///
    /// Receives the calculations of other senders and returns the
    /// updated total physical potential energy.
    #[heavy_computation]
    #[efficient_alternatives(
        "update_energy_set_changed_forces_with_receivers",
        "update_energy_add_changed_forces_with_receivers"
    )]
    fn update_energy_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<T, <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Sets the forces of this group after a change in the position of a single atom.
    #[efficient_alternatives(
        "update_energy_set_changed_forces_with_senders",
        "update_energy_set_changed_forces_with_receivers"
    )]
    fn set_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Adds the forces arising from this potential to the forces of this group
    /// after a change in the position of a single atom.
    #[efficient_alternatives(
        "update_energy_add_changed_forces_with_senders",
        "update_energy_add_changed_forces_with_receivers"
    )]
    fn add_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, AS, MS, AR, MR>>::Error>;
}
