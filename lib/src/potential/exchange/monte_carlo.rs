use super::ExchangePotential;
use crate::core::{
    AtomGroup,
    sync_ops::{SyncAddReceiver, SyncAddSender, SyncMulReceiver, SyncMulSender},
};
use macros::{efficient_alternatives, heavy_computation};
use std::sync::{
    Barrier, RwLock,
    mpsc::{Receiver, Sender},
};

/// An enum for tracking relations between images.
#[derive(Clone, Copy, Debug)]
pub enum NeighboringImage {
    /// The current image.
    This,
    /// This image's predecessor.
    ///
    /// For the first image, the last one counts as its predecessor.
    Prev,
    /// This image's successor.
    ///
    /// For the last image, the first on counts as its successor.
    Next,
}

/// A trait for exchange potentials that may be used in a Monte-Carlo algorithm.
pub trait MonteCarloExchangePotential<T, V, AS, MS, AR, MR>:
    ExchangePotential<T, V, AS, MS, AR, MR>
where
    AS: SyncAddSender<T> + ?Sized,
    MS: SyncMulSender<T> + ?Sized,
    AR: SyncAddReceiver<T> + ?Sized,
    MR: SyncMulReceiver<T> + ?Sized,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image and sets the forces of this group in this image accordingly.
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
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        type_positions_last_image: &[AtomGroup<V>],
        type_positions_next_image: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image and sets the forces of this group in this image accordingly.
    ///
    /// Receives the calculations of other senders and returns the
    /// updated total exhange potential energy.
    #[heavy_computation]
    fn update_energy_set_changed_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        type_positions_last_image: &[AtomGroup<V>],
        type_positions_next_image: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces_group: &mut [V],
    ) -> Result<T, <Self as MonteCarloExchangePotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image
    /// and adds the forces arising from this potential to the forces of this group in this image.
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
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        type_positions_last_image: &[AtomGroup<V>],
        type_positions_next_image: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image
    /// and adds the forces arising from this potential to the forces of this group in this image.
    ///
    /// Receives the calculations of other senders and returns the
    /// updated total exhange potential energy.
    #[heavy_computation]
    fn update_energy_add_changed_forces_with_receivers(
        &mut self,
        channel: &Receiver<T>,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut AR,
        multiplier: &mut MR,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        type_positions_last_image: &[AtomGroup<V>],
        type_positions_next_image: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces_group: &mut [V],
    ) -> Result<T, <Self as MonteCarloExchangePotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image.
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
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        type_positions_last_image: &[AtomGroup<V>],
        type_positions_next_image: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image.
    ///
    /// Receives the calculations of other senders and returns the
    /// updated total exhange potential energy.
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
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        type_positions_last_image: &[AtomGroup<V>],
        type_positions_next_image: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
    ) -> Result<T, <Self as MonteCarloExchangePotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Sets the forces of this group in this image after a change
    /// in the position of a single atom in either a neighboring or this image.
    #[heavy_computation]
    #[efficient_alternatives("update_energy_set_changed_forces")]
    fn set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        type_positions_last_image: &[AtomGroup<V>],
        type_positions_next_image: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V, AS, MS, AR, MR>>::Error>;

    /// Adds the forces arising from this potential to the forces of this group in this image
    /// after a change in the position of a single atom in either a neighboring or this image.
    #[heavy_computation]
    #[efficient_alternatives("update_energy_add_changed_forces")]
    fn add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        type_positions_last_image: &[AtomGroup<V>],
        type_positions_next_image: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces_group: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V, AS, MS, AR, MR>>::Error>;
}
