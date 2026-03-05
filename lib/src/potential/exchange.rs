//! Traits for updating the forces and calculating the exchange potential energy.

use crate::core::marker::{InnerIsLeading, InnerIsTrailing};

#[cfg(feature = "monte_carlo")]
mod monte_carlo;
pub mod quadratic;

/// A trait for exchange potentials that yield the contribution of a group in the first image
/// to the total exchange potential energy of a given type of atoms.
pub trait LeadingExchangePotential<T, V> {
    /// Calculates the contribution of this group in the first image to the total exchange potential energy
    /// of the type and sets the forces of this group in the first image accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group in the first image to the total exchange potential energy
    /// of the type and adds the forces arising from this potential
    /// to the forces of this group in the first image accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group in the first image to the total exchange potential energy
    /// of the type.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
    ) -> T;

    /// Sets the forces of this group in the first image.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    );

    /// Adds the forces arising from this potential to the forces of this group in the first image.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    );
}

/// A trait for exchange potentials that yield the contribution of a group in an inner image
/// to the total exchange potential energy of a given type of atoms.
pub trait InnerExchangePotential<T, V> {
    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type and sets the forces of this group in this image accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type and adds the forces arising from this potential
    /// to the forces of this group in this image accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group in this image to the total exchange potential energy
    /// of the type.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
    ) -> T;

    /// Sets the forces of this group in this image.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    );

    /// Adds the forces arising from this potential to the forces of this group in this image.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    );
}

/// A trait for exchange potentials that yield the contribution of a group in the last image
/// to the total exchange potential energy of a given type of atoms.
pub trait TrailingExchangePotential<T, V> {
    /// Calculates the contribution of this group in the last image to the total exchange potential energy
    /// of the type and sets the forces of this group in the last image accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group in the last image to the total exchange potential energy
    /// of the type and adds the forces arising from this potential
    /// to the forces of this group in the last image accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this group in the last image to the total exchange potential energy
    /// of the type.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
    ) -> T;

    /// Sets the forces of this group in the last image.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    );

    /// Adds the forces arising from this potential to the forces of this group in the last image.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    );
}

impl<T, V, U> LeadingExchangePotential<T, V> for U
where
    U: InnerExchangePotential<T, V> + InnerIsLeading + ?Sized,
{
    fn calculate_potential_set_forces(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_set_forces(
            self,
            type_positions_last_image,
            type_positions_next_image,
            type_positions,
            group_forces,
        )
    }

    fn calculate_potential_add_forces(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_add_forces(
            self,
            type_positions_last_image,
            type_positions_next_image,
            type_positions,
            group_forces,
        )
    }

    fn calculate_potential(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerExchangePotential::calculate_potential(
            self,
            type_positions_last_image,
            type_positions_next_image,
            type_positions,
        )
    }

    fn set_forces(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::set_forces(
            self,
            type_positions_last_image,
            type_positions_next_image,
            type_positions,
            group_forces,
        );
    }

    fn add_forces(
        &mut self,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::add_forces(
            self,
            type_positions_last_image,
            type_positions_next_image,
            type_positions,
            group_forces,
        );
    }
}

impl<T, V, U> TrailingExchangePotential<T, V> for U
where
    U: InnerExchangePotential<T, V> + InnerIsTrailing + ?Sized,
{
    fn calculate_potential_set_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_set_forces(
            self,
            type_positions_prev_image,
            type_positions_first_image,
            type_positions,
            group_forces,
        )
    }

    fn calculate_potential_add_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_add_forces(
            self,
            type_positions_prev_image,
            type_positions_first_image,
            type_positions,
            group_forces,
        )
    }

    fn calculate_potential(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerExchangePotential::calculate_potential(
            self,
            type_positions_prev_image,
            type_positions_first_image,
            type_positions,
        )
    }

    fn set_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::set_forces(
            self,
            type_positions_prev_image,
            type_positions_first_image,
            type_positions,
            group_forces,
        );
    }

    fn add_forces(
        &mut self,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::add_forces(
            self,
            type_positions_prev_image,
            type_positions_first_image,
            type_positions,
            group_forces,
        );
    }
}

#[cfg(feature = "monte_carlo")]
pub use monte_carlo::{
    InnerMonteCarloExchangePotential, LeadingMonteCarloExchangePotential, TrailingMonteCarloExchangePotential,
};
