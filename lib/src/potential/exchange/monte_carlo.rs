use super::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential};
use crate::core::marker::{InnerIsLeading, InnerIsTrailing};

#[derive(Clone, Copy, Debug)]
pub enum NeighboringImage {
    This,
    Prev,
    Next,
}

/// A trait for exchange potentials of the first image
/// that may be used in a Monte-Carlo algorithm.
pub trait LeadingMonteCarloExchangePotential<T, V>: LeadingExchangePotential<T, V> {
    /// Calculates the contribution of this group in the first image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or the first image
    /// and updates the forces of this group in the first image accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group in the first image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or the first image
    /// and adds the updated forces to the forces of this group in the first image.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group in the first image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or the first image.
    ///
    /// Returns the contribution to the change in total energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
    ) -> Result<T, Self::Error>;

    /// Updates the forces of this group in the first image after a change
    /// in the position of a single atom in either a neighboring or the first image.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Adds the updated forces to the forces of this group in the first image given a change
    /// in the position of a single atom in either a neighboring or the first image.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;
}

/// A trait for exchange potentials of an inner image
/// that may be used in a Monte-Carlo algorithm.
pub trait InnerMonteCarloExchangePotential<T, V>: InnerExchangePotential<T, V> {
    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image
    /// and updates the forces of this group in this image accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this grop in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image
    /// and adds the updated forces to the forces of this group in this image.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group in this image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or this image.
    ///
    /// Returns the contribution to the change in total energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
    ) -> Result<T, Self::Error>;

    /// Updates the forces of this group in this image after a change
    /// in the position of a single atom in either a neighboring or this image.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Adds the updated forces to the forces of this group in this image given a change
    /// in the position of a single atom in either a neighboring or this image.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;
}

/// A trait for exchange potentials of the last image
/// that may be used in a Monte-Carlo algorithm.
pub trait TrailingMonteCarloExchangePotential<T, V>: TrailingExchangePotential<T, V> {
    /// Calculates the contribution of this group in the last image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or the last image
    /// and updates the forces of this group in the last image accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group in the last image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or the last image
    /// and adds the updated forces to the forces of this group in the last image.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error>;

    /// Calculates the contribution of this group in the last image to the change in total exchange
    /// potential energy of the type after a change in the position of a single atom
    /// in either a neighboring or the last image.
    ///
    /// Returns the contribution to the change in total energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
    ) -> Result<T, Self::Error>;

    /// Updates the forces of this group in the last image after a change
    /// in the position of a single atom in either a neighboring or the last image.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Adds the updated forces to the forces of this group in the last image given a change
    /// in the position of a single atom in either a neighboring or the last image.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error>;
}

impl<T, V, U> LeadingMonteCarloExchangePotential<T, V> for U
where
    U: InnerMonteCarloExchangePotential<T, V> + InnerIsLeading + ?Sized,
{
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        type_positions_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        InnerMonteCarloExchangePotential::calculate_potential_diff_set_changed_forces(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_prev_image,
            type_positions_next_image,
            type_positions,
            group_forces,
        )
    }

    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        position_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        InnerMonteCarloExchangePotential::calculate_potential_diff_add_changed_forces(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_last_image,
            position_next_image,
            type_positions,
            group_forces,
        )
    }

    fn calculate_potential_diff(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        position_next_image: &[V],
        type_positions: &[V],
    ) -> Result<T, Self::Error> {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::calculate_potential_diff(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_last_image,
            position_next_image,
            type_positions,
        )
    }

    fn set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        position_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::set_changed_forces(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_last_image,
            position_next_image,
            type_positions,
            group_forces,
        )
    }

    fn add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_last_image: &[V],
        position_next_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::add_changed_forces(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_last_image,
            position_next_image,
            type_positions,
            group_forces,
        )
    }
}

impl<T, V, U> TrailingMonteCarloExchangePotential<T, V> for U
where
    U: InnerMonteCarloExchangePotential<T, V> + InnerIsTrailing + ?Sized,
{
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        position_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        InnerMonteCarloExchangePotential::calculate_potential_diff_set_changed_forces(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_prev_image,
            position_first_image,
            type_positions,
            group_forces,
        )
    }

    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        position_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        InnerMonteCarloExchangePotential::calculate_potential_diff_add_changed_forces(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_prev_image,
            position_first_image,
            type_positions,
            group_forces,
        )
    }

    fn calculate_potential_diff(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        position_first_image: &[V],
        type_positions: &[V],
    ) -> Result<T, Self::Error> {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::calculate_potential_diff(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_prev_image,
            position_first_image,
            type_positions,
        )
    }

    fn set_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        position_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::set_changed_forces(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_prev_image,
            position_first_image,
            type_positions,
            group_forces,
        )
    }

    fn add_changed_forces(
        &mut self,
        changed_image: NeighboringImage,
        changed_atom_index: usize,
        old_value: V,
        type_positions_prev_image: &[V],
        position_first_image: &[V],
        type_positions: &[V],
        group_forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::add_changed_forces(
            self,
            changed_image,
            changed_atom_index,
            old_value,
            type_positions_prev_image,
            position_first_image,
            type_positions,
            group_forces,
        )
    }
}
