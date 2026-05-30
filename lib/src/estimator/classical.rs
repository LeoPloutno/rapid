//! Traits for calculating classical observables.

use crate::{
    core::{
        GroupInTypeInImageInSystem, Scheme,
        stat::{Bosonic, Distinguishable},
        sync_ops::{SyncAddReciever, SyncAddSender, SyncMulReciever, SyncMulSender},
    },
    potential::{
        exchange::{ExchangePotential, quadratic::QuadraticExpansionExchangePotential},
        physical::PhysicalPotential,
    },
};

mod atom_additive;
pub use atom_additive::{
    AtomAdditiveClassicalEstimatorReciever, AtomAdditiveClassicalEstimatorSender,
    AtomAdditiveMinimalClassicalEstimatorSender,
};
mod atom_multiplicative;
pub use atom_multiplicative::{
    AtomMultiplicativeClassicalEstimatorReciever, AtomMultiplicativeClassicalEstimatorSender,
    AtomMultiplicativeMinimalClassicalEstimatorSender,
};

/// A wrapper for implementors of the [`MinimalClassicalEstimatorSender`] trait.
pub struct MinimalClassicalEstimator<E: ?Sized>(pub(crate) E);

impl<E> MinimalClassicalEstimator<E> {
    /// Wraps the provided value with `MinimalClassicalEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

/// A trait for classical estimators that recieve
/// the calculations of classical estimator senders
/// and outut the final value.
pub trait ClassicalEstimatorReciever<T, V, Adder, Multiplier>
where
    Adder: SyncAddReciever<Self::Output> + ?Sized,
    Multiplier: SyncMulReciever<Self::Output> + ?Sized,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the observable.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
    ) -> Result<Self::Output, Self::Error>;
}

/// A trait for classical estimators.
pub trait ClassicalEstimatorSender<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group to the observable
    /// and sends it to a [`ClassicalEstimatorReciever`].
    ///
    /// Assumes this group obeys distinguishable statistics.
    fn calculate_distinguishable(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Dist, &mut DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group to the observable
    /// and sends it to a [`ClassicalEstimatorReciever`].
    ///
    /// Assumes this group obeys bosonic statistics.
    fn calculate_bosonic(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Boson, &mut BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error>;
}

/// A trait for classical estimators that do not rely on either
/// the physical nor the exchange potentials.
pub trait MinimalClassicalEstimatorSender<T, V, Adder, Multiplier>
where
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group to the observable
    /// and sends it to a [`ClassicalEstimatorReciever`].
    ///
    /// Assumes this group obeys distinguishable statistics.
    fn calculate_distinguishable(
        &mut self,
        exchange_potential_is_cyclic: bool,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group to the observable
    /// and sends it to a [`ClassicalEstimatorReciever`].
    ///
    /// Assumes this group obeys bosonic statistics.
    fn calculate_bosonic(
        &mut self,
        exchange_potential_is_cyclic: bool,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error>;
}

impl<T, V, Adder, Multiplier, E> MinimalClassicalEstimatorSender<T, V, Adder, Multiplier>
    for MinimalClassicalEstimator<E>
where
    Adder: SyncAddSender<E::Output> + ?Sized,
    Multiplier: SyncMulSender<E::Output> + ?Sized,
    E: MinimalClassicalEstimatorSender<T, V, Adder, Multiplier> + ?Sized,
{
    type Output = E::Output;
    type Error = E::Error;

    fn calculate_distinguishable(
        &mut self,
        exchange_potential_is_cyclic: bool,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error> {
        self.0.calculate_distinguishable(
            exchange_potential_is_cyclic,
            adder,
            multiplier,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            group_heat,
            group_kinetic_energy,
            positions,
            momenta,
            physical_forces,
            exchange_forces,
        )
    }

    fn calculate_bosonic(
        &mut self,
        exchange_potential_is_cyclic: bool,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error> {
        self.0.calculate_bosonic(
            exchange_potential_is_cyclic,
            adder,
            multiplier,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            group_heat,
            group_kinetic_energy,
            positions,
            momenta,
            physical_forces,
            exchange_forces,
        )
    }
}

impl<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad, E>
    ClassicalEstimatorSender<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad>
    for MinimalClassicalEstimator<E>
where
    Adder: SyncAddSender<<Self as MinimalClassicalEstimatorSender<T, V, Adder, Multiplier>>::Output>
        + ?Sized,
    Multiplier: SyncMulSender<<Self as MinimalClassicalEstimatorSender<T, V, Adder, Multiplier>>::Output>
        + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: MinimalClassicalEstimatorSender<T, V, Adder, Multiplier>,
{
    type Output = <Self as MinimalClassicalEstimatorSender<T, V, Adder, Multiplier>>::Output;
    type Error = <Self as MinimalClassicalEstimatorSender<T, V, Adder, Multiplier>>::Error;

    fn calculate_distinguishable(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Dist, &mut DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error> {
        MinimalClassicalEstimatorSender::calculate_distinguishable(
            self,
            exchange_potential.is_cyclic(),
            adder,
            multiplier,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            group_heat,
            group_kinetic_energy,
            positions,
            momenta,
            physical_forces,
            exchange_forces,
        )
    }

    fn calculate_bosonic(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Boson, &mut BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error> {
        MinimalClassicalEstimatorSender::calculate_bosonic(
            self,
            exchange_potential.is_cyclic(),
            adder,
            multiplier,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            group_heat,
            group_kinetic_energy,
            positions,
            momenta,
            physical_forces,
            exchange_forces,
        )
    }
}
