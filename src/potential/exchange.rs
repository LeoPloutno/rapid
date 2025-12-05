use crate::core::AtomGroupInfo;
use crate::marker::{InnerIsLeading, InnerIsTrailing};

/// A trait for exchange potentials that yield the contribution of the first replica
/// to the total exchange potential energy of a given group.
pub trait LeadingExchangePotential<T, V> {
    /// Calculates the contribution of the first replica to the total exchange potential energy
    /// of the group and sets the forces of this group in the first replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the first replica to the total exchange potential energy
    /// of the group and adds the forces arising from this potential
    /// to the forces of this group in the first replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the first replica to the total exchange potential energy
    /// of the group.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
    ) -> T;

    /// Sets the forces of this group in the first replica.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    );

    /// Adds the forces arising from this potential to the forces of this group in the first replica.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    );
}

/// A trait for exchange potentials that yield the contribution of an inner replica
/// to the total exchange potential energy of a given group.
pub trait InnerExchangePotential<T, V> {
    /// Calculates the contribution of this replica to the total exchange potential energy
    /// of the group and sets the forces of this group in this replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this replica to the total exchange potential energy
    /// of the group and adds the forces arising from this potential
    /// to the forces of this group in this replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this replica to the total exchange potential energy
    /// of the group.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
    ) -> T;

    /// Sets the forces of this group in this replica.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    );

    /// Adds the forces arising from this potential to the forces of this group in this replica.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    );
}

/// A trait for exchange potentials that yield the contribution of the last replica
/// to the total exchange potential energy of a given group.
pub trait TrailingExchangePotential<T, V> {
    /// Calculates the contribution of the last replica to the total exchange potential energy
    /// of the group and sets the forces of this group in the last replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the last replica to the total exchange potential energy
    /// of the group and adds the forces arising from this potential
    /// to the forces of this group in the last replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the last replica to the total exchange potential energy
    /// of the group.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
    ) -> T;

    /// Sets the forces of this group in the last replica.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    );

    /// Adds the forces arising from this potential to the forces of this group in the last replica.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    );
}

impl<T, V, U> LeadingExchangePotential<T, V> for U
where
    U: InnerExchangePotential<T, V> + InnerIsLeading,
{
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_set_forces(
            self,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
            forces,
        )
    }

    fn calculate_potential_add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_add_forces(
            self,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
            forces,
        )
    }

    fn calculate_potential(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerExchangePotential::calculate_potential(
            self,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
        )
    }

    fn set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::set_forces(
            self,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
            forces,
        );
    }

    fn add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::add_forces(
            self,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
            forces,
        );
    }
}

impl<T, V, U> TrailingExchangePotential<T, V> for U
where
    U: InnerExchangePotential<T, V> + InnerIsTrailing,
{
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_set_forces(
            self,
            group,
            last_replica,
            positions_prev_replica,
            position_first_replica,
            positions,
            forces,
        )
    }

    fn calculate_potential_add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_add_forces(
            self,
            group,
            last_replica,
            positions_prev_replica,
            position_first_replica,
            positions,
            forces,
        )
    }

    fn calculate_potential(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerExchangePotential::calculate_potential(
            self,
            group,
            last_replica,
            positions_prev_replica,
            position_first_replica,
            positions,
        )
    }

    fn set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::set_forces(
            self,
            group,
            last_replica,
            positions_prev_replica,
            position_first_replica,
            positions,
            forces,
        );
    }

    fn add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::add_forces(
            self,
            group,
            last_replica,
            positions_prev_replica,
            position_first_replica,
            positions,
            forces,
        );
    }
}

pub mod monte_carlo;

pub mod normal_modes;

#[cfg(feature = "monte_carlo")]
pub use monte_carlo::{
    InnerMonteCarloExchangePotential, LeadingMonteCarloExchangePotential,
    TrailingMonteCarloExchangePotential,
};
