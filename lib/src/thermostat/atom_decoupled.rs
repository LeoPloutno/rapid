/// A trait for thermostats that decouple all atoms from each
/// other such that each one can be thermalized independently.
///
/// Any implementor of this trait automatically implements [`Thermostat`]
/// if the associated error type is convertible from [`EmptyIteratorError`].
///
/// [`Thermostat`]: super::Thermostat
/// [`EmptyIteratorError`]: crate::core::EmptyIteratorError
pub trait AtomDecoupledThermostat<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Thermalizes the atom.
    ///
    /// Returns the contribution of this atom to the change
    /// in the internal energy of the system due to thermalization.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn thermalize(
        &mut self,
        step_size: T,
        atom_index: usize,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
        momentum: &mut V,
    ) -> Result<T, Self::Error>;
}
