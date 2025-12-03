use crate::core::AtomGroupInfo;

/// A trait for physical potentials that can be expressed as a sum
/// of potentials that depend only on a singlee atom at a time.
///
/// Any implementor of this trait automatically implements [`GroupDecoupledPhysicalPotential`].
///
/// [`GroupDecoupledPhysicalPotential`]: super::GroupDecoupledPhysicalPotential
pub trait AtomDecoupledPhysicalPotential<T, V> {
    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the replica and sets the force of this atom accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_force(
        &mut self,
        idx: usize,
        group: &AtomGroupInfo<T>,
        position: &V,
        force: &mut V,
    ) -> T;

    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the replica and adds the force arising from this potential to the force of this atom.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_force(
        &mut self,
        idx: usize,
        group: &AtomGroupInfo<T>,
        position: &V,
        force: &mut V,
    ) -> T;

    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the replica.
    ///
    /// Returns the contribution to the total energy.
    #[deprecated = "Consider using `calculate_potential_set_force` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(&mut self, idx: usize, group: &AtomGroupInfo<T>, position: &V) -> T;

    /// Sets the force of this atom.
    #[deprecated = "Consider using `calculate_potential_set_force` as a more efficient alternative"]
    fn set_force(&mut self, idx: usize, group: &AtomGroupInfo<T>, position: &V, force: &mut V);

    /// Adds the force arising from this potential to the force of this atom.
    #[deprecated = "Consider using `calculate_potential_add_force` as a more efficient alternative"]
    fn add_force(&mut self, idx: usize, group: &AtomGroupInfo<T>, position: &V, force: &mut V);
}

#[cfg(feature = "monte_carlo")]
pub(super) mod monte_carlo {
    use super::AtomDecoupledPhysicalPotential;
    use crate::core::AtomGroupInfo;

    /// A trait for atom-deoupled physical potentials that may be used in a Monte-Carlo algorithm.
    ///
    /// Any implementor of this trait automatically implements [`GroupDecoupledPhysicalPotential`].
    ///
    /// [`MonteCarloGroupDecoupledPhysicalPotential`]: super::MonteCarloGroupDecoupledPhysicalPotential
    pub trait MonteCarloAtomDecoupledPhysicalPotential<T, V>:
        AtomDecoupledPhysicalPotential<T, V>
    {
        /// Calculates the change in the physical potential energy of this atom
        /// after a change in its position and updates the force of this atom accordingly.
        ///
        /// Returns the change in potential energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff_set_changed_force(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            position: &V,
            force: &mut V,
        ) -> T;

        /// Calculates the change in the physical potential energy of this atom
        /// after a change in its position and adds the updated force to the force
        /// of this atom.
        ///
        /// Returns the change in potential energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff_add_changed_force(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            position: &V,
            force: &mut V,
        ) -> T;

        /// Calculates the change in the physical potential energy of this atom
        /// after a change in its position.
        ///
        /// Returns the change in potential energy.
        #[deprecated = "Consider using `calculate_potential_diff_set_changed_force` as a more efficient alternative"]
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            position: &V,
        ) -> T;

        /// Updates the force of this atom after a change to its position.
        #[deprecated = "Consider using `calculate_potential_diff_set_changed_force` as a more efficient alternative"]
        fn set_changed_force(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            position: &V,
            force: &mut V,
        );

        /// Adds the updated force to the force of this atom given a change in its position.
        #[deprecated = "Consider using `calculate_potential_diff_add_changed_force` as a more efficient alternative"]
        fn add_changed_force(
            &mut self,
            changed_position_idx: usize,
            old_value: &V,
            group: &AtomGroupInfo<T>,
            position: &V,
            force: &mut V,
        );
    }
}
