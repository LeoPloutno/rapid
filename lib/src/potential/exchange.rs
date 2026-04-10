//! Traits for updating the forces and calculating the exchange potential energy.

use macros::{efficient_alternatives, heavy_computation};

pub mod quadratic;

#[cfg(feature = "monte_carlo")]
mod monte_carlo;
#[cfg(feature = "monte_carlo")]
pub use monte_carlo::{MonteCarloExchangePotential, NeighboringImage};

/// A trait for exchange potentials.
pub trait ExchangePotential<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Returns whether this exchange potential is invariant under
    /// a cyclic permutation of the images.
    fn is_cyclic(&self) -> bool;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type and sets the forces of this group accordingly.
    ///
    /// Returns the contribution to the total exchange potential energy.
    #[heavy_computation]
    fn calculate_potential_set_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type and adds the forces arising from this potential to the forces of this group.
    ///
    /// Returns the contribution to the total exchange potential energy.
    #[heavy_computation]
    fn calculate_potential_add_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type.
    ///
    /// Returns the contribution to the total exchange potential energy.
    #[heavy_computation]
    #[efficient_alternatives("calculate_potential_set_forces", "calculate_potential_add_forces")]
    fn calculate_potential(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
    ) -> Result<T, Self::Error>;

    /// Sets the forces of this group in this image.
    #[efficient_alternatives("calculate_potential_set_forces")]
    fn set_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Adds the forces arising from this potential to the forces of this group in this image.
    #[efficient_alternatives("calculate_potential_add_forces")]
    fn add_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;
}
