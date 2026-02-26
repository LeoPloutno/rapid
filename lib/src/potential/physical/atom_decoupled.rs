#[cfg(feature = "monte_carlo")]
mod monte_carlo;

/// A trait for physical potentials that can be expressed as a sum
/// of potentials that depend only on a single atom.
///
/// Any implementor of this trait automatically implements [`PhysicalPotential`].
///
/// [`PhysicalPotential`]: super::PhysicalPotential
pub trait AtomDecoupledPhysicalPotential<T, V> {
    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the image and sets the force of this atom accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_force(&mut self, atom_index: usize, position: &V, force: &mut V) -> T;

    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the image and adds the force arising from this potential to the force of this atom.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_force(&mut self, atom_index: usize, position: &V, force: &mut V) -> T;

    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the image.
    ///
    /// Returns the contribution to the total energy.
    /// - `position`: The position of this atom.
    #[deprecated = "Consider using `calculate_potential_set_force` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(&mut self, atom_index: usize, position: &V) -> T;

    /// Sets the force of this atom.
    #[deprecated = "Consider using `calculate_potential_set_force` as a more efficient alternative"]
    fn set_force(&mut self, atom_index: usize, position: &V, force: &mut V);

    /// Adds the force arising from this potential to the force of this atom.
    #[deprecated = "Consider using `calculate_potential_add_force` as a more efficient alternative"]
    fn add_force(&mut self, atom_index: usize, position: &V, force: &mut V);
}

#[cfg(feature = "monte_carlo")]
pub use monte_carlo::MonteCarloAtomDecoupledPhysicalPotential;
