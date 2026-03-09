//! A trait for thermalizing the system.

use std::ops::Add;

use arc_rw_lock::ElementRwLock;

use crate::{ImageHandle, core::error::EmptyIteratorError, zip_items, zip_iterators};

mod atom_decoupled;

/// A trait for thermostats.
///
/// A thermostat is an entity that thermalized a system
/// in the canonical ensemble such that different energies
/// are sampled while keeping the temperature fixed.
pub trait Thermostat<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Performs thermalization of the system.
    ///
    /// Returns the contribution of this group in this image to the
    /// change in the internal energy of the system due to thermalization.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn thermalize(
        &mut self,
        step_size: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
        group_momenta: &mut [V],
    ) -> Result<T, Self::Error>;
}

impl<T, V, U> Thermostat<T, V> for U
where
    T: Clone + Add<Output = T>,
    U: AtomDecoupledThermostat<T, V, Error: From<EmptyIteratorError>>,
{
    type Error = <Self as AtomDecoupledThermostat<T, V>>::Error;

    fn thermalize(
        &mut self,
        step_size: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
        group_momenta: &mut [V],
    ) -> Result<T, Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read(),
            group_momenta
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, physical_force, exchange_force, momentum))| {
                AtomDecoupledThermostat::thermalize(
                    self,
                    step_size.clone(),
                    index,
                    position,
                    physical_force,
                    exchange_force,
                    momentum,
                )
            },
        );
        let first_atom_energy_diff = iter.next().ok_or(EmptyIteratorError)??;
        iter.try_fold(first_atom_energy_diff, |accum_energy_diff, atom_energy_diff| {
            Ok(accum_energy_diff + atom_energy_diff?)
        })
    }
}

pub use atom_decoupled::AtomDecoupledThermostat;
