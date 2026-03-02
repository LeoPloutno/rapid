use crate::{
    core::{GroupImageHandle, GroupTypeHandle},
    marker::{InnerIsLeading, InnerIsTrailing},
    sync_ops::{SyncAddRecv, SyncAddSend, SyncMulRecv, SyncMulSend},
};

/// A trait for quantum estimators.
/// The implementor of this trait recieves the calculations of
/// the other quantum observables and produces an output.
pub trait MainQuantumObservable<T, V, Adder, Multiplier>
where
    Adder: SyncAddRecv<T> + ?Sized,
    Multiplier: SyncMulRecv<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the observable.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(&mut self, adder: &mut Adder, multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error>;
}

/// A trait for quantum estimators operating in the first replica for a specific group.
pub trait LeadingQuantumObservable<T, V, Adder, Multiplier>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in the first replica
    /// to the observable and sends it to a `MainQuantumObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantum estimators operating in an inner replica for a specific group.
pub trait InnerQuantumObservable<T, V, Adder, Multiplier>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in this replica
    /// to the observable and sends it to a `MainQuantumObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantum estimators operating in the last replica for a specific group.
pub trait TrailingQuantumObservable<T, V, Adder, Multiplier>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in the last replica
    /// to the observable and sends it to a `MainQuantumObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

impl<T, V, Adder, Multiplier, U> LeadingQuantumObservable<T, V, Adder, Multiplier> for U
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    U: InnerQuantumObservable<T, V, Adder, Multiplier> + InnerIsLeading + ?Sized,
{
    type Output = <Self as InnerQuantumObservable<T, V, Adder, Multiplier>>::Output;
    type Error = <Self as InnerQuantumObservable<T, V, Adder, Multiplier>>::Error;

    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error> {
        InnerQuantumObservable::calculate(
            self,
            adder,
            multiplier,
            physical_potential_energy,
            exchange_potential_energy,
            groups_positions,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}

impl<T, V, Adder, Multiplier, U> TrailingQuantumObservable<T, V, Adder, Multiplier> for U
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    U: InnerQuantumObservable<T, V, Adder, Multiplier> + InnerIsTrailing + ?Sized,
{
    type Output = <Self as InnerQuantumObservable<T, V, Adder, Multiplier>>::Output;
    type Error = <Self as InnerQuantumObservable<T, V, Adder, Multiplier>>::Error;

    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error> {
        InnerQuantumObservable::calculate(
            self,
            adder,
            multiplier,
            physical_potential_energy,
            exchange_potential_energy,
            groups_positions,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}
