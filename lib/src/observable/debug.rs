use crate::{
    core::{GroupImageHandle, GroupTypeHandle},
    marker::{InnerIsLeading, InnerIsTrailing},
    sync_ops::{SyncAddRecv, SyncAddSend, SyncMulRecv, SyncMulSend},
};

/// A trait for quantities which may be used to debug the simulation.
/// The implementor of this trait recieves the calculations of
/// the other debug observables and produces an output.
pub trait MainDebugObservable<T, V, Adder, Multiplier>
where
    Adder: SyncAddRecv<T> + ?Sized,
    Multiplier: SyncMulRecv<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the quantity.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(&mut self, adder: &mut Adder, multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error>;
}

/// A trait for quantities which may be used to debug the simulation,
/// operating in the first replica for a specific group of atoms.
pub trait LeadingDebugObservable<T, V, Adder, Multiplier>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in the first replica
    /// to the quantity and sends it to a `MainDebugObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantities which may be used to debug the simulation,
/// operating in an inner replica for a specific group of atoms.
pub trait InnerDebugObservable<T, V, Adder, Multiplier>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in this replica
    /// to the quantity and sends it to a `MainDebugObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantities which may be used to debug the simulation,
/// operating in the last replica for a specific group.
pub trait TrailingDebugObservable<T, V, Adder, Multiplier>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in the last replica
    /// to the quantity and sends it to a `MainDebugObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

impl<T, V, Adder, Multiplier, U> LeadingDebugObservable<T, V, Adder, Multiplier> for U
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    U: InnerDebugObservable<T, V, Adder, Multiplier> + InnerIsLeading + ?Sized,
{
    type Output = <Self as InnerDebugObservable<T, V, Adder, Multiplier>>::Output;
    type Error = <Self as InnerDebugObservable<T, V, Adder, Multiplier>>::Error;

    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error> {
        InnerDebugObservable::calculate(
            self,
            adder,
            multiplier,
            physical_potential_energy,
            exchange_potential_energy,
            kinetic_energy,
            groups_positions,
            groups_momenta,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}

impl<T, V, Adder, Multiplier, U> TrailingDebugObservable<T, V, Adder, Multiplier> for U
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    U: InnerDebugObservable<T, V, Adder, Multiplier> + InnerIsTrailing + ?Sized,
{
    type Output = <Self as InnerDebugObservable<T, V, Adder, Multiplier>>::Output;
    type Error = <Self as InnerDebugObservable<T, V, Adder, Multiplier>>::Error;

    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error> {
        InnerDebugObservable::calculate(
            self,
            adder,
            multiplier,
            physical_potential_energy,
            exchange_potential_energy,
            kinetic_energy,
            groups_positions,
            groups_momenta,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}
