use super::PhysicalPotential;
use crate::core::{GroupInTypeInImage, ValidOutput, monte_carlo::ChangedGroup};
use macros::{efficient_alternatives, heavy_computation};

/// A trait for physical potentials that may be used in a Monte-Carlo algorithm.
///
/// The generic parameter `O` is the type of the values returned by the energy calculations.
/// Setting it to `()` implies that the calculations are sent to another potential
/// that combines the recieved data and returns the total physical potential energy.
pub trait MonteCarloPhysicalPotential<T, V, O: ValidOutput<T>>: PhysicalPotential<T, V, O> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of a group to the total physical potential energy
    /// of the image after a change in the position of a single atom
    /// and sets the forces of that group accordingly.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_new_energy_set_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<O, <Self as MonteCarloPhysicalPotential<T, V, O>>::Error>;

    /// Calculates the contribution of a group to the total physical potential energy
    /// of the image after a change in the position of a single atom
    /// and adds the forces arising from this potential to the forces of that group.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    fn calculate_new_energy_add_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<O, <Self as MonteCarloPhysicalPotential<T, V, O>>::Error>;

    /// Calculates the contribution of a group to the total physical potential energy
    /// of the image after a change in the position of a single atom.
    ///
    /// Where applicable, returns the potential energy.
    #[heavy_computation]
    #[efficient_alternatives(
        "calculate_new_energy_set_changed_forces",
        "calculate_new_energy_add_changed_forces"
    )]
    fn calculate_new_energy(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_energy: T,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<O, <Self as MonteCarloPhysicalPotential<T, V, O>>::Error>;

    /// Sets the forces of a group after a change in the position of a single atom.
    #[efficient_alternatives("calculate_new_energy_set_changed_forces")]
    fn set_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, O>>::Error>;

    /// Adds the forces arising from this potential to the forces of a group
    /// after a change in the position of a single atom.
    #[efficient_alternatives("calculate_new_energy_add_changed_forces")]
    fn add_changed_forces(
        &mut self,
        changed_group: ChangedGroup,
        changed_atom_index: usize,
        old_position: V,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), <Self as MonteCarloPhysicalPotential<T, V, O>>::Error>;
}
