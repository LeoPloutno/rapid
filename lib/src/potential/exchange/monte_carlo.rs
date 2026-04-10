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
pub trait MonteCarloExchangePotential<T, V>: ExchangePotential<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image and sets the forces of this group in this image accordingly.
    ///
    /// Returns the contribution to the change in total exchange potential energy.
    #[heavy_computation]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloExchangePotential<T, V>>::Error>;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image
    /// and adds the forces arising from this potential to the forces of this group in this image.
    ///
    /// Returns the contribution to the change in total exchange potential energy.
    #[heavy_computation]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, <Self as MonteCarloExchangePotential<T, V>>::Error>;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image.
    ///
    /// Returns the contribution to the change in total exchange potential energy.
    #[heavy_computation]
    #[efficient_alternatives(
        "calculate_potential_diff_set_changed_forces",
        "calculate_potential_diff_add_changed_forces"
    )]
    fn calculate_potential_diff(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
    ) -> Result<T, <Self as MonteCarloExchangePotential<T, V>>::Error>;

    /// Sets the forces of this group in this image after a change
    /// in the position of a single atom in either a neighboring or this image.
    #[heavy_computation]
    #[efficient_alternatives("calculate_potential_diff_set_changed_forces")]
    fn set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V>>::Error>;

    /// Adds the forces arising from this potential to the forces of this group in this image
    /// after a change in the position of a single atom in either a neighboring or this image.
    #[heavy_computation]
    #[efficient_alternatives("calculate_potential_diff_add_changed_forces")]
    fn add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloExchangePotential<T, V>>::Error>;
}
