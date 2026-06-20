//! Traits for updating the forces and calculating the physical potential energy.

use std::sync::{
    Barrier, RwLock,
    mpsc::{Receiver, Sender},
};

use crate::core::sync_ops::{SyncAddReceiver, SyncAddSender, SyncMulReceiver, SyncMulSender};

use super::GroupInTypeInImage;
use macros::{efficient_alternatives, heavy_computation};

mod atom_additive;
pub use atom_additive::AtomAdditivePhysicalPotential;

#[cfg(feature = "monte_carlo")]
mod monte_carlo;

#[cfg(feature = "monte_carlo")]
pub use self::{
    atom_additive::AtomAdditiveMonteCarloPhysicalPotential,
    monte_carlo::MonteCarloPhysicalPotential,
};

/// A trait for physical potentials.
pub trait PhysicalPotential<T, V, AS, MS, AR, MR>
where
    AS: SyncAddSender<T> + ?Sized,
    MS: SyncMulSender<T> + ?Sized,
    AR: SyncAddReceiver<T> + ?Sized,
    MR: SyncMulReceiver<T> + ?Sized,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the image and sets the forces of this group accordingly.
    ///
    /// Sends the results of the calculations to a receiver.
    #[heavy_computation]
    fn calculate_energy_set_forces_with_senders(
        &mut self,
        channel: &Sender<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AS,
        multiplier: &mut MS,
        positions: &GroupInTypeInImage<V>,
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the image and sets the forces of this group accordingly.
    ///
    /// Receives the calculations of other senders and returns the total
    /// physical potential energy.
    #[heavy_computation]
    fn calculate_energy_set_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        positions: &GroupInTypeInImage<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the image and adds the forces arising from this potential to the forces of this group.
    ///
    /// Sends the results of the calculations to a receiver.
    #[heavy_computation]
    fn calculate_energy_add_forces_with_senders(
        &mut self,
        channel: &Sender<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AS,
        multiplier: &mut MS,
        positions: &GroupInTypeInImage<V>,
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the image and adds the forces arising from this potential to the forces of this group.
    ///
    /// Receives the calculations of other senders and returns the total
    /// physical potential energ.
    #[heavy_computation]
    fn calculate_energy_add_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        positions: &GroupInTypeInImage<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the image.
    ///
    /// Sends the results of the calculations to a receiver.
    #[heavy_computation]
    #[efficient_alternatives(
        "calculate_energy_set_forces_with_senders",
        "calculate_energy_add_forces_with_senders"
    )]
    fn calculate_energy_with_senders(
        &mut self,
        channel: &Sender<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AS,
        multiplier: &mut MS,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group to the total physical potential energy
    /// of the image.
    ///
    /// Receives the calculations of other senders and returns the total
    /// physical potential energy.
    #[heavy_computation]
    #[efficient_alternatives(
        "calculate_energy_set_forces_with_receivers",
        "calculate_energy_add_forces_with_receivers"
    )]
    fn calculate_energy_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<T, Self::Error>;

    /// Sets the forces of this group.
    #[efficient_alternatives(
        "calculate_energy_set_forces_with_senders",
        "calculate_energy_set_forces_with_receivers"
    )]
    fn set_forces(
        &mut self,
        positions: &GroupInTypeInImage<V>,
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Adds the forces arising from this potential to the forces of this group.
    #[efficient_alternatives(
        "calculate_energy_add_forces_with_senders",
        "calculate_energy_add_forces_with_receivers"
    )]
    fn add_forces(
        &mut self,
        positions: &GroupInTypeInImage<V>,
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;
}
