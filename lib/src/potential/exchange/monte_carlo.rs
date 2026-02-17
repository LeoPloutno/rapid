use super::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential};
use crate::marker::{InnerIsLeading, InnerIsTrailing};

#[derive(Clone, Copy, Debug)]
pub enum NeighboringReplica {
    This,
    Prev,
    Next,
}

/// A trait for exchange potentials of the first replica
/// that may be used in a Monte-Carlo algorithm.
pub trait LeadingMonteCarloExchangePotential<T, V>: LeadingExchangePotential<T, V> {
    /// Calculates the contribution of the first replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or the first replica
    /// and updates the group_forces of this group in the first replica accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_last_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the first replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or the first replica
    /// and adds the updated group_forces to the group_forces of this group in the first replica.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_last_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the first replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or the first replica.
    ///
    /// Returns the contribution to the change in total energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_last_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
    ) -> T;

    /// Updates the group_forces of this group in the first replica after a change
    /// in the position of a single atom in either a neighboring or the first replica.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_last_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Adds the updated group_forces to the group_forces of this group in the first replica given a change
    /// in the position of a single atom in either a neighboring or the first replica.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_last_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;
}

/// A trait for exchange potentials of an inner replica
/// that may be used in a Monte-Carlo algorithm.
pub trait InnerMonteCarloExchangePotential<T, V>: InnerExchangePotential<T, V> {
    /// Calculates the contribution of this replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or this replica
    /// and updates the group_forces of this group in this accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or this replica
    /// and adds the updated group_forces to the group_forces of this group in this replica.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or this replica.
    ///
    /// Returns the contribution to the change in total energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
    ) -> T;

    /// Updates the group_forces of this group in this replica after a change
    /// in the position of a single atom in either a neighboring or this replica.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Adds the updated group_forces to the group_forces of this group in this replica given a change
    /// in the position of a single atom in either a neighboring or this replica.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;
}

/// A trait for exchange potentials of the last replica
/// that may be used in a Monte-Carlo algorithm.
pub trait TrailingMonteCarloExchangePotential<T, V>: TrailingExchangePotential<T, V> {
    /// Calculates the contribution of the last replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or the last replica
    /// and updates the group_forces of this group in the last replica accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the last replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or the last replica
    /// and adds the updated group_forces to the group_forces of this group in the last replica.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the last replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or the last replica.
    ///
    /// Returns the contribution to the change in total energy.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_first_replica: &[V],
        group_positions: &[V],
    ) -> T;

    /// Updates the group_forces of this group in the last replica after a change
    /// in the position of a single atom in either a neighboring or the last replica.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;

    /// Adds the updated group_forces to the group_forces of this group in the last replica given a change
    /// in the position of a single atom in either a neighboring or the last replica.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T;
}

impl<T, V, U> LeadingMonteCarloExchangePotential<T, V> for U
where
    U: InnerMonteCarloExchangePotential<T, V> + InnerIsLeading,
{
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        group_positions_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerMonteCarloExchangePotential::calculate_potential_diff_set_changed_forces(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_prev_replica,
            group_positions_next_replica,
            group_positions,
            group_forces,
        )
    }

    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_last_replica: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerMonteCarloExchangePotential::calculate_potential_diff_add_changed_forces(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_last_replica,
            position_next_replica,
            group_positions,
            group_forces,
        )
    }

    fn calculate_potential_diff(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_last_replica: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::calculate_potential_diff(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_last_replica,
            position_next_replica,
            group_positions,
        )
    }

    fn set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_last_replica: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::set_changed_forces(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_last_replica,
            position_next_replica,
            group_positions,
            group_forces,
        )
    }

    fn add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_last_replica: &[V],
        position_next_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::add_changed_forces(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_last_replica,
            position_next_replica,
            group_positions,
            group_forces,
        )
    }
}

impl<T, V, U> TrailingMonteCarloExchangePotential<T, V> for U
where
    U: InnerMonteCarloExchangePotential<T, V> + InnerIsTrailing,
{
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerMonteCarloExchangePotential::calculate_potential_diff_set_changed_forces(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_prev_replica,
            position_first_replica,
            group_positions,
            group_forces,
        )
    }

    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        InnerMonteCarloExchangePotential::calculate_potential_diff_add_changed_forces(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_prev_replica,
            position_first_replica,
            group_positions,
            group_forces,
        )
    }

    fn calculate_potential_diff(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::calculate_potential_diff(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_prev_replica,
            position_first_replica,
            group_positions,
        )
    }

    fn set_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::set_changed_forces(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_prev_replica,
            position_first_replica,
            group_positions,
            group_forces,
        )
    }

    fn add_changed_forces(
        &mut self,
        changed_replica: NeighboringReplica,
        changed_atom_idx: usize,
        old_value: V,
        group_positions_prev_replica: &[V],
        position_first_replica: &[V],
        group_positions: &[V],
        group_forces: &mut [V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::add_changed_forces(
            self,
            changed_replica,
            changed_atom_idx,
            old_value,
            group_positions_prev_replica,
            position_first_replica,
            group_positions,
            group_forces,
        )
    }
}
