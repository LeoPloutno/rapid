//! Traits for updating the forces and calculating the exchange potential energy.

use super::GroupInTypeInImage;
use crate::core::{
    AtomGroup,
    sync_ops::{SyncAddReceiver, SyncAddSender, SyncMulReceiver, SyncMulSender},
};
use macros::{efficient_alternatives, heavy_computation};
use std::sync::{
    Barrier, RwLock,
    mpsc::{Receiver, Sender},
};

pub mod quadratic;

#[cfg(feature = "monte_carlo")]
mod monte_carlo;
#[cfg(feature = "monte_carlo")]
pub use monte_carlo::{MonteCarloExchangePotential, NeighboringImage};

/// A trait for exchange potentials.
pub trait ExchangePotential<T, V, AS, MS, AR, MR>
where
    AS: SyncAddSender<T> + ?Sized,
    MS: SyncMulSender<T> + ?Sized,
    AR: SyncAddReceiver<T> + ?Sized,
    MR: SyncMulReceiver<T> + ?Sized,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Returns whether this exchange potential is invariant under
    /// a cyclic permutation of the images.
    fn is_cyclic(&self) -> bool;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type and sets the forces of this group accordingly.
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
        positions_prev_image: &GroupInTypeInImage<V>,
        positions_next_image: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type and sets the forces of this group accordingly.
    ///
    /// Receives the calculations of other senders and returns the total
    /// exchange potential energy.
    #[heavy_computation]
    fn calculate_energy_set_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        positions_prev_image: &GroupInTypeInImage<V>,
        positions_next_image: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type and adds the forces arising from this potential to the forces of this group.
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
        positions_prev_image: &GroupInTypeInImage<V>,
        positions_next_image: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type and adds the forces arising from this potential to the forces of this group.
    ///
    /// Receives the calculations of other senders and returns the total
    /// exchange potential energy.
    #[heavy_computation]
    fn calculate_energy_add_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        positions_prev_image: &GroupInTypeInImage<V>,
        positions_next_image: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type.
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
        positions_prev_image: &GroupInTypeInImage<V>,
        positions_next_image: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type.
    ///
    /// Receives the calculations of other senders and returns the total
    /// exchange potential energy.
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
        positions_prev_image: &GroupInTypeInImage<V>,
        positions_next_image: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<T, Self::Error>;

    /// Sets the forces of this group in this image.
    #[efficient_alternatives("calculate_energy_set_forces")]
    fn set_forces(
        &mut self,
        positions_prev_image: &GroupInTypeInImage<V>,
        positions_next_image: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [AtomGroup<V>],
    ) -> Result<(), Self::Error>;

    /// Adds the forces arising from this potential to the forces of this group in this image.
    #[efficient_alternatives("calculate_energy_add_forces")]
    fn add_forces(
        &mut self,
        positions_prev_image: &GroupInTypeInImage<V>,
        positions_next_image: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces_group: &mut [V],
    ) -> Result<(), Self::Error>;
}
