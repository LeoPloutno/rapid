use super::ExchangePotential;
use crate::core::{AtomGroup, marker::ValidOutput};
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
///
/// The generic parameter `O` is the type of the values returned by the energy calculations.
/// Setting it to `()` implies that the calculations are sent to another potential
/// that combines the recieved data and returns the total exchange potential energy.
pub trait MonteCarloExchangePotential<T, V, O: ValidOutput<T>>: ExchangePotential<T, V, O> {
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
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: T,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces: &mut [V],
    ) -> Result<O, <Self as MonteCarloExchangePotential<T, V, O>>::Error>;

    /// Calculates the contribution of a group to the total exchange potential energy
    /// of the type after a change in the position of a single atom
    /// and adds the forces arising from this potential to the forces of that group.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_new_energy_add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces: &mut [V],
    ) -> Result<O, <Self as MonteCarloExchangePotential<T, V, O>>::Error>;

    /// Calculates the contribution of a group to the total exchange potential energy
    /// of the type after a change in the position of a single atom.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    #[efficient_alternatives(
        "calculate_new_energy_set_changed_forces",
        "calculate_new_energy_add_changed_forces"
    )]
    fn calculate_new_energy(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
    ) -> Result<O, <Self as MonteCarloExchangePotential<T, V, O>>::Error>;

    /// Sets the forces of a group after a change
    /// in the position of a single atom in either a neighboring or this image.
    #[heavy_computation]
    #[efficient_alternatives("calculate_new_energy_set_changed_forces")]
    fn set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V, O>>::Error>;

    /// Adds the forces arising from this potential to the forces of a group
    /// after a change in the position of a single atom in either a neighboring or this image.
    #[heavy_computation]
    #[efficient_alternatives("calculate_new_energy_add_changed_forces")]
    fn add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        prev_image_type_positions: &[AtomGroup<V>],
        next_image_type_positions: &[AtomGroup<V>],
        type_positions: &[AtomGroup<V>],
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V, O>>::Error>;
}
