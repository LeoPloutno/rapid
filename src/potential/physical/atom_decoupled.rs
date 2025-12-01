use crate::{core::AtomGroupInfo, vector::Vector};

/// A trait for physical potentials that can be expressed as a sum
/// of potentials that depend only on a singlee atom at a time.
pub trait AtomDecoupledPhysicalPotential<T, const N: usize, V>
where
    V: Vector<N, Element = T>,
{
    /// Calculates the contribution of this atom to the total potential energy
    /// of the replica and sets the force of this atom accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_atom_potential_set_force(
        &mut self,
        idx: usize,
        group: &AtomGroupInfo<T>,
        position: &V,
        force: &mut V,
    ) -> T;

    /// Calculates the contribution of this atom to the total potential energy
    /// of the replica.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_atom_potential_set_force` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_atom_potential(&mut self, idx: usize, group: &AtomGroupInfo<T>, position: &V)
    -> T;

    /// Sets the force of this atom.
    #[deprecated = "Consider using `calculate_atom_potential_set_force` as a more efficient alternative"]
    fn set_atom_force(&mut self, idx: usize, group: &AtomGroupInfo<T>, position: &V, force: &mut V);
}

#[cfg(feature = "monte_carlo")]
pub(super) mod monte_carlo {
    use super::AtomDecoupledPhysicalPotential;
    use crate::{core::AtomGroupInfo, vector::Vector};

    /// A trait for atom-deoupled physical potentials that may be used in a Monte-Carlo algorithm.
    pub trait MonteCarloAtomDecoupledPhysicalPotential<T, const N: usize, V>:
        AtomDecoupledPhysicalPotential<T, N, V>
    where
        V: Vector<N, Element = T>,
    {
        /// Calculates the change in the potential energy of this atom
        /// after a change in its position and sets the force of this atom accordingly.
        ///
        /// Returns the change in potential energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_atom_potential_diff_update_force(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            position: &V,
            force: &mut V,
        ) -> T;

        /// Calculates the change in the potential energy of this atom
        /// after a change in its position.
        ///
        /// Returns the change in potential energy.
        #[deprecated = "Consider using `calculate_atom_potential_diff_update_force` as a more efficient alternative"]
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_atom_potential_diff(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            position: &V,
        ) -> T;

        /// Updates the force of this atom after a change to its position.
        #[deprecated = "Consider using `calculate_atom_potential_diff_update_force` as a more efficient alternative"]
        fn update_atom_force(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            position: &V,
            force: &mut V,
        );
    }
}
