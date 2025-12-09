use crate::core::AtomGroupInfo;

#[cfg(feature = "monte_carlo")]
mod monte_carlo;

/// A trait for physical potentials that can be expressed as a sum
/// of potentials that depend only on a singlee atom at a time.
///
/// Any implementor of this trait automatically implements [`GroupDecoupledPhysicalPotential`].
///
/// [`GroupDecoupledPhysicalPotential`]: super::GroupDecoupledPhysicalPotential
pub trait AtomDecoupledPhysicalPotential<T, V> {
    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the replica and sets the atom_force of this atom accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_force(
        &mut self,
        group: &AtomGroupInfo<T>,
        atom_idx: usize,
        atom_position: &V,
        atom_force: &mut V,
    ) -> T;

    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the replica and adds the atom_force arising from this potential to the atom_force of this atom.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_force(
        &mut self,
        group: &AtomGroupInfo<T>,
        atom_idx: usize,
        atom_position: &V,
        atom_force: &mut V,
    ) -> T;

    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the replica.
    ///
    /// Returns the contribution to the total energy.
    /// - `atom_position`: The atom_position of this atom.
    #[deprecated = "Consider using `calculate_potential_set_force` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        group: &AtomGroupInfo<T>,
        atom_idx: usize,
        atom_position: &V,
    ) -> T;

    /// Sets the atom_force of this atom.
    #[deprecated = "Consider using `calculate_potential_set_force` as a more efficient alternative"]
    fn set_force(
        &mut self,
        group: &AtomGroupInfo<T>,
        atom_idx: usize,
        atom_position: &V,
        atom_force: &mut V,
    );

    /// Adds the atom_force arising from this potential to the atom_force of this atom.
    #[deprecated = "Consider using `calculate_potential_add_force` as a more efficient alternative"]
    fn add_force(
        &mut self,
        group: &AtomGroupInfo<T>,
        atom_idx: usize,
        atom_position: &V,
        atom_force: &mut V,
    );
}

#[cfg(feature = "monte_carlo")]
pub use monte_carlo::MonteCarloAtomDecoupledPhysicalPotential;
