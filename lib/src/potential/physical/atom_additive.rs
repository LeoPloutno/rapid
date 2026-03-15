use std::ops::Add;

use crate::{
    ImageHandle,
    core::{Additive as AdditivePhysicalPotential, error::EmptyError},
    potential::physical::PhysicalPotential,
    zip_items, zip_iterators,
};

#[cfg(feature = "monte_carlo")]
mod monte_carlo;
#[cfg(feature = "monte_carlo")]
pub use monte_carlo::AtomAdditiveMonteCarloPhysicalPotential;

/// A trait for physical potentials that can be expressed as a sum
/// of potentials that depend only on a single atom.
///
/// For any type `P` that implements this trait, [`AdditivePhysicalPotential<P>`]
/// atomatically implements [`PhysicalPotential`].
pub trait AtomAdditivePhysicalPotential<T: Add<Output = T>, V> {
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditivePhysicalPotential<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<EmptyError>;

    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the image and sets the force of this atom accordingly.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_force(
        &mut self,
        atom_index: usize,
        position: &V,
        force: &mut V,
    ) -> Result<T, Self::ErrorAtom>;

    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the image and adds the force arising from this potential to the force of this atom.
    ///
    /// Returns the contribution to the total energy.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_add_force(
        &mut self,
        atom_index: usize,
        position: &V,
        force: &mut V,
    ) -> Result<T, Self::ErrorAtom>;

    /// Calculates the contribution of this atom to the total physical potential energy
    /// of the image.
    ///
    /// Returns the contribution to the total energy.
    /// - `position`: The position of this atom.
    #[deprecated = "Consider using `calculate_potential_set_force` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(&mut self, atom_index: usize, position: &V) -> Result<T, Self::ErrorAtom>;

    /// Sets the force of this atom.
    #[deprecated = "Consider using `calculate_potential_set_force` as a more efficient alternative"]
    fn set_force(&mut self, atom_index: usize, position: &V, force: &mut V) -> Result<(), Self::ErrorAtom>;

    /// Adds the force arising from this potential to the force of this atom.
    #[deprecated = "Consider using `calculate_potential_add_force` as a more efficient alternative"]
    fn add_force(&mut self, atom_index: usize, position: &V, force: &mut V) -> Result<(), Self::ErrorAtom>;
}

impl<T, V, P> AtomAdditivePhysicalPotential<T, V> for AdditivePhysicalPotential<P>
where
    T: Add<Output = T>,
    P: AtomAdditivePhysicalPotential<T, V> + ?Sized,
{
    type ErrorAtom = P::ErrorAtom;
    type ErrorSystem = P::ErrorSystem;

    fn calculate_potential_set_force(
        &mut self,
        atom_index: usize,
        position: &V,
        force: &mut V,
    ) -> Result<T, Self::ErrorAtom> {
        self.0.calculate_potential_set_force(atom_index, position, force)
    }

    fn calculate_potential_add_force(
        &mut self,
        atom_index: usize,
        position: &V,
        force: &mut V,
    ) -> Result<T, Self::ErrorAtom> {
        self.0.calculate_potential_add_force(atom_index, position, force)
    }

    fn calculate_potential(&mut self, atom_index: usize, position: &V) -> Result<T, Self::ErrorAtom> {
        #[allow(deprecated)]
        self.0.calculate_potential(atom_index, position)
    }

    fn set_force(&mut self, atom_index: usize, position: &V, force: &mut V) -> Result<(), Self::ErrorAtom> {
        #[allow(deprecated)]
        self.0.set_force(atom_index, position, force)
    }

    fn add_force(&mut self, atom_index: usize, position: &V, force: &mut V) -> Result<(), Self::ErrorAtom> {
        #[allow(deprecated)]
        self.0.add_force(atom_index, position, force)
    }
}

impl<T, V, P> PhysicalPotential<T, V> for AdditivePhysicalPotential<P>
where
    T: Add<Output = T>,
    P: ?Sized,
    Self: AtomAdditivePhysicalPotential<T, V>,
{
    type Error = <Self as AtomAdditivePhysicalPotential<T, V>>::ErrorSystem;

    fn calculate_potential_set_forces(
        &mut self,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        let mut iter = zip_iterators!(groups_positions.read().read(), group_forces)
            .enumerate()
            .map(|(index, zip_items!(position, force))| {
                AtomAdditivePhysicalPotential::calculate_potential_set_force(self, index, position, force)
            });
        let first_atom_potential_energy = iter.next().ok_or(EmptyError)??;
        Ok(iter.try_fold(
            first_atom_potential_energy,
            |accum_potential_energy, atom_potential_energy| {
                Ok::<_, <Self as AtomAdditivePhysicalPotential<T, V>>::ErrorAtom>(
                    accum_potential_energy + atom_potential_energy?,
                )
            },
        )?)
    }

    fn calculate_potential_add_forces(
        &mut self,
        groups_positions: &ImageHandle<V>,
        group_forces: &mut [V],
    ) -> Result<T, Self::Error> {
        let mut iter = zip_iterators!(groups_positions.read().read(), group_forces)
            .enumerate()
            .map(|(index, zip_items!(position, force))| {
                AtomAdditivePhysicalPotential::calculate_potential_add_force(self, index, position, force)
            });
        let first_atom_potential_energy = iter.next().ok_or(EmptyError)??;
        Ok(iter.try_fold(
            first_atom_potential_energy,
            |accum_potential_energy, atom_potential_energy| {
                Ok::<_, <Self as AtomAdditivePhysicalPotential<T, V>>::ErrorAtom>(
                    accum_potential_energy + atom_potential_energy?,
                )
            },
        )?)
    }

    fn calculate_potential(&mut self, groups_positions: &ImageHandle<V>) -> Result<T, Self::Error> {
        let mut iter = groups_positions
            .read()
            .read()
            .iter()
            .enumerate()
            .map(|(index, position)| {
                #[allow(deprecated)]
                AtomAdditivePhysicalPotential::calculate_potential(self, index, position)
            });
        let first_atom_potential_energy = iter.next().ok_or(EmptyError)??;
        Ok(iter.try_fold(
            first_atom_potential_energy,
            |accum_potential_energy, atom_potential_energy| {
                Ok::<_, <Self as AtomAdditivePhysicalPotential<T, V>>::ErrorAtom>(
                    accum_potential_energy + atom_potential_energy?,
                )
            },
        )?)
    }

    fn set_forces(&mut self, groups_positions: &ImageHandle<V>, group_forces: &mut [V]) -> Result<(), Self::Error> {
        for (index, zip_items!(position, force)) in
            zip_iterators!(groups_positions.read().read(), group_forces).enumerate()
        {
            #[allow(deprecated)]
            AtomAdditivePhysicalPotential::set_force(self, index, position, force)?;
        }
        Ok(())
    }

    fn add_forces(&mut self, groups_positions: &ImageHandle<V>, group_forces: &mut [V]) -> Result<(), Self::Error> {
        for (index, zip_items!(position, force)) in
            zip_iterators!(groups_positions.read().read(), group_forces).enumerate()
        {
            #[allow(deprecated)]
            AtomAdditivePhysicalPotential::add_force(self, index, position, force)?;
        }
        Ok(())
    }
}
