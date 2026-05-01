//! Traits for thermostats that can thermalize atoms separately.

use super::{GroupInTypeInImageInSystem, Thermostat};
use crate::{
    core::{Decoupled as DecoupledThermostat, error::EmptyError},
    zip_items, zip_iterators,
};
use macros::heavy_computation;
use std::ops::Add;

/// A trait for thermostats that decouple all atoms from each
/// other such that each one can be thermalized independently.
///
/// For any type `T` that implements this trait, [`AtomDecoupledThermostat<T>`]
/// atomatically implements [`Thermostat`].
pub trait AtomDecoupledThermostat<T, V>
where
    T: Add<Output = T>,
{
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomDecoupledThermostat<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<EmptyError>;

    /// Thermalizes the atom.
    ///
    /// Returns the contribution of this atom to the change
    /// in the internal energy of the system due to thermalization.
    #[heavy_computation]
    fn thermalize(
        &mut self,
        atom_index: usize,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
        momentum: &mut V,
    ) -> Result<T, Self::ErrorAtom>;
}

impl<T, V, U> AtomDecoupledThermostat<T, V> for DecoupledThermostat<U>
where
    T: Clone + Add<Output = T>,
    U: AtomDecoupledThermostat<T, V> + ?Sized,
{
    type ErrorAtom = U::ErrorAtom;
    type ErrorSystem = U::ErrorSystem;

    fn thermalize(
        &mut self,
        atom_index: usize,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
        momentum: &mut V,
    ) -> Result<T, Self::ErrorAtom> {
        self.0.thermalize(
            atom_index,
            position,
            physical_force,
            exchange_force,
            momentum,
        )
    }
}

impl<T, V, U> Thermostat<T, V> for DecoupledThermostat<U>
where
    T: Add<Output = T>,
    U: ?Sized,
    Self: AtomDecoupledThermostat<T, V>,
{
    type Error = <Self as AtomDecoupledThermostat<T, V>>::ErrorSystem;

    fn thermalize(
        &mut self,
        positions: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
        group_momenta: &mut [V],
    ) -> Result<T, Self::Error> {
        let mut iter = zip_iterators!(positions, physical_forces, exchange_forces, group_momenta)
            .enumerate()
            .map(
                |(index, zip_items!(position, physical_force, exchange_force, momentum))| {
                    AtomDecoupledThermostat::thermalize(
                        self,
                        index,
                        position,
                        physical_force,
                        exchange_force,
                        momentum,
                    )
                },
            );
        let first_atom_heat = iter.next().ok_or(EmptyError)??;
        Ok(iter.try_fold(first_atom_heat, |accum_heat, atom_heat| {
            Ok::<_, <Self as AtomDecoupledThermostat<T, V>>::ErrorAtom>(accum_heat + atom_heat?)
        })?)
    }
}
