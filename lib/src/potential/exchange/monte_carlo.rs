use std::sync::{Barrier, RwLock};

use crate::core::{AtomGroup, ValidOutput};

use super::ExchangePotential;
use macros::{efficient_alternatives, heavy_computation};

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
pub trait MonteCarloExchangePotential<T, V, A, M, O>: ExchangePotential<T, V, A, M, O>
where
    A: ?Sized,
    M: ?Sized,
    O: ValidOutput<T>,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of a group to the total exchange potential energy
    /// of the type after a change in the position of a single atom
    /// and sets the forces of that group accordingly.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_new_energy_set_changed_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: T,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces: &mut [V],
    ) -> Result<O, <Self as MonteCarloExchangePotential<T, V, A, M, O>>::Error>;

    /// Calculates the contribution of a group to the total exchange potential energy
    /// of the type after a change in the position of a single atom
    /// and adds the forces arising from this potential to the forces of that group.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_new_energy_add_changed_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces: &mut [V],
    ) -> Result<O, <Self as MonteCarloExchangePotential<T, V, A, M, O>>::Error>;

    /// Calculates the contribution of a group to the total exchange potential energy
    /// of the type after a change in the position of a single atom.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    #[efficient_alternatives(
        "calculate_new_energy_set_changed_forces",
        "calculate_new_energy_add_changed_forces"
    )]
    fn calculate_potential_diff(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
    ) -> Result<O, <Self as MonteCarloExchangePotential<T, V, A, M, O>>::Error>;

    /// Sets the forces of a group after a change
    /// in the position of a single atom in either a neighboring or this image.
    #[heavy_computation]
    #[efficient_alternatives("calculate_new_energy_set_changed_forces")]
    fn set_changed_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V, A, M, O>>::Error>;

    /// Adds the forces arising from this potential to the forces of a group
    /// after a change in the position of a single atom in either a neighboring or this image.
    #[heavy_computation]
    #[efficient_alternatives("calculate_new_energy_add_changed_forces")]
    fn add_changed_forces(
        &mut self,
        barrier: &Barrier,
        shared_value: &RwLock<T>,
        adder: &mut A,
        multiplier: &mut M,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V, A, M, O>>::Error>;
}
