use crate::core::AtomGroupInfo;
use crate::marker::{InnerIsLeading, InnerIsTrailing};

#[cfg(feature = "monte_carlo")]
mod monte_carlo;
pub mod quadratic;

/// A trait for exchange potentials that yield the contribution of the first replica
/// to the total exchange potential energy of a given group.
pub trait LeadingExchangePotential<T, V> {
    /// Calculates the contribution of the first replica to the total exchange potential energy
    /// of the group and sets the group_forces of this group in the first replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the first replica to the total exchange potential energy
    /// of the group and adds the group_forces arising from this potential
    /// to the group_forces of this group in the first replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
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
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
    ) -> T;

    /// Sets the group_forces of this group in the first replica.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    );

    /// Adds the group_forces arising from this potential to the group_forces of this group in the first replica.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    );
}

/// A trait for exchange potentials that yield the contribution of an inner replica
/// to the total exchange potential energy of a given group.
pub trait InnerExchangePotential<T, V> {
    /// Calculates the contribution of this replica to the total exchange potential energy
    /// of the group and sets the group_forces of this group in this replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this replica to the total exchange potential energy
    /// of the group and adds the group_forces arising from this potential
    /// to the group_forces of this group in this replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this replica to the total exchange potential energy
    /// of the group.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
    ) -> T;

    /// Sets the group_forces of this group in this replica.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    );

    /// Adds the group_forces arising from this potential to the group_forces of this group in this replica.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    );
}

/// A trait for exchange potentials that yield the contribution of the last replica
/// to the total exchange potential energy of a given group.
pub trait TrailingExchangePotential<T, V> {
    /// Calculates the contribution of the last replica to the total exchange potential energy
    /// of the group and sets the group_forces of this group in the last replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the last replica to the total exchange potential energy
    /// of the group and adds the group_forces arising from this potential
    /// to the group_forces of this group in the last replica accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_forces(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the last replica to the total exchange potential energy
    /// of the group.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
    ) -> T;

    /// Sets the group_forces of this group in the last replica.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    );

    /// Adds the group_forces arising from this potential to the group_forces of this group in the last replica.
    #[deprecated = "Consider using `calculate_potential_add_forces` as a more efficient alternative"]
    fn add_forces(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    );
}

impl<T, V, U> LeadingExchangePotential<T, V> for U
where
    U: InnerExchangePotential<T, V> + InnerIsLeading,
{
    fn calculate_potential_set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_set_forces(
            self,
            0,
            group,
            last_replica_group_positions,
            position_next_replica,
            group_positions,
            group_forces,
        )
    }

    fn calculate_potential_add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_add_forces(
            self,
            0,
            group,
            last_replica_group_positions,
            position_next_replica,
            group_positions,
            group_forces,
        )
    }

    fn calculate_potential(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerExchangePotential::calculate_potential(
            self,
            0,
            group,
            last_replica_group_positions,
            position_next_replica,
            group_positions,
        )
    }

    fn set_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::set_forces(
            self,
            0,
            group,
            last_replica_group_positions,
            position_next_replica,
            group_positions,
            group_forces,
        );
    }

    fn add_forces(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica_group_positions: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::add_forces(
            self,
            0,
            group,
            last_replica_group_positions,
            position_next_replica,
            group_positions,
            group_forces,
        );
    }
}

impl<T, V, U> TrailingExchangePotential<T, V> for U
where
    U: InnerExchangePotential<T, V> + InnerIsTrailing,
{
    fn calculate_potential_set_forces(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_set_forces(
            self,
            last_replica,
            group,
            prev_replica_group_positions,
            position_first_replica,
            group_positions,
            group_forces,
        )
    }

    fn calculate_potential_add_forces(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerExchangePotential::calculate_potential_add_forces(
            self,
            last_replica,
            group,
            prev_replica_group_positions,
            position_first_replica,
            group_positions,
            group_forces,
        )
    }

    fn calculate_potential(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerExchangePotential::calculate_potential(
            self,
            last_replica,
            group,
            prev_replica_group_positions,
            position_first_replica,
            group_positions,
        )
    }

    fn set_forces(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::set_forces(
            self,
            last_replica,
            group,
            prev_replica_group_positions,
            position_first_replica,
            group_positions,
            group_forces,
        );
    }

    fn add_forces(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        prev_replica_group_positions: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) {
        #[allow(deprecated)]
        InnerExchangePotential::add_forces(
            self,
            last_replica,
            group,
            prev_replica_group_positions,
            position_first_replica,
            group_positions,
            group_forces,
        );
    }
}

#[cfg(feature = "monte_carlo")]
pub use monte_carlo::{
    InnerMonteCarloExchangePotential, LeadingMonteCarloExchangePotential,
    TrailingMonteCarloExchangePotential,
};
