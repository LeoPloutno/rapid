use super::{
    InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential,
    marker::{InnerIsLeading, InnerIsTrailing},
};
use crate::core::AtomGroupInfo;

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
    /// and updates the forces of this group in the first replica accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the first replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or the first replica
    /// and adds the updated forces to the forces of this group in the first replica.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
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
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
    ) -> T;

    /// Updates the forces of this group in the first replica after a change
    /// in the position of a single atom in either a neighboring or the first replica.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Adds the updated forces to the forces of this group in the first replica given a change
    /// in the position of a single atom in either a neighboring or the first replica.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;
}

/// A trait for exchange potentials of an inner replica
/// that may be used in a Monte-Carlo algorithm.
pub trait InnerMonteCarloExchangePotential<T, V>: InnerExchangePotential<T, V> {
    /// Calculates the contribution of this replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or this replica
    /// and updates the forces of this group in this accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of this replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or this replica
    /// and adds the updated forces to the forces of this group in this replica.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
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
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
    ) -> T;

    /// Updates the forces of this group in this replica after a change
    /// in the position of a single atom in either a neighboring or this replica.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Adds the updated forces to the forces of this group in this replica given a change
    /// in the position of a single atom in either a neighboring or this replica.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;
}

/// A trait for exchange potentials of the last replica
/// that may be used in a Monte-Carlo algorithm.
pub trait TrailingMonteCarloExchangePotential<T, V>: TrailingExchangePotential<T, V> {
    /// Calculates the contribution of the last replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or the last replica
    /// and updates the forces of this group in the last replica accordingly.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Calculates the contribution of the last replica to the change in total exchange
    /// potential energy of this group after a change in the position of a single atom
    /// in either a neighboring or the last replica
    /// and adds the updated forces to the forces of this group in the last replica.
    ///
    /// Returns the contribution to the change in total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
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
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
    ) -> T;

    /// Updates the forces of this group in the last replica after a change
    /// in the position of a single atom in either a neighboring or the last replica.
    #[deprecated = "Consider using `calculate_potential_diff_set_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;

    /// Adds the updated forces to the forces of this group in the last replica given a change
    /// in the position of a single atom in either a neighboring or the last replica.
    #[deprecated = "Consider using `calculate_potential_diff_add_changed_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T;
}

impl<T, V, U> LeadingMonteCarloExchangePotential<T, V> for U
where
    U: InnerMonteCarloExchangePotential<T, V> + InnerIsLeading,
{
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        InnerMonteCarloExchangePotential::calculate_potential_diff_set_changed_forces(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
            forces,
        )
    }

    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        InnerMonteCarloExchangePotential::calculate_potential_diff_add_changed_forces(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
            forces,
        )
    }

    fn calculate_potential_diff(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::calculate_potential_diff(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
        )
    }

    fn set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::set_changed_forces(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
            forces,
        )
    }

    fn add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        positions_last_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::add_changed_forces(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            0,
            positions_last_replica,
            position_next_replica,
            positions,
            forces,
        )
    }
}

impl<T, V, U> TrailingMonteCarloExchangePotential<T, V> for U
where
    U: InnerMonteCarloExchangePotential<T, V> + InnerIsTrailing,
{
    fn calculate_potential_diff_set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        InnerMonteCarloExchangePotential::calculate_potential_diff_set_changed_forces(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            replica,
            positions_prev_replica,
            position_first_replica,
            positions,
            forces,
        )
    }

    fn calculate_potential_diff_add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        InnerMonteCarloExchangePotential::calculate_potential_diff_add_changed_forces(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            replica,
            positions_prev_replica,
            position_first_replica,
            positions,
            forces,
        )
    }

    fn calculate_potential_diff(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::calculate_potential_diff(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            replica,
            positions_prev_replica,
            position_first_replica,
            positions,
        )
    }

    fn set_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::set_changed_forces(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            replica,
            positions_prev_replica,
            position_first_replica,
            positions,
            forces,
        )
    }

    fn add_changed_forces(
        &mut self,
        changed_atom_idx: usize,
        changed_replica: NeighboringReplica,
        old_value: &V,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_prev_replica: &[V],
        position_first_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> T {
        #[allow(deprecated)]
        InnerMonteCarloExchangePotential::add_changed_forces(
            self,
            changed_atom_idx,
            changed_replica,
            old_value,
            group,
            replica,
            positions_prev_replica,
            position_first_replica,
            positions,
            forces,
        )
    }
}
