//! Traits for calculating quantum observables.

use crate::{
    core::{
        AtomGroup, AtomTypeReaderLock, MapInWhole, MapOutsideWhole, Scheme,
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
    AtomAdditiveMinimalQuantumEstimatorSender, AtomAdditiveQuantumEstimatorReciever,
    AtomAdditiveQuantumEstimatorSender,
};
mod atom_multiplicative;
pub use atom_multiplicative::{
    AtomMultiplicativeMinimalQuantumEstimatorSender, AtomMultiplicativeQuantumEstimatorReciever,
    AtomMultiplicativeQuantumEstimatorSender,
};

mod estimator_images {
    use std::ops::Deref;

    /// Holds information about the images relevant to calculations of observables.
    #[derive(Clone, Copy, Debug)]
    pub enum EstimatorImages<T> {
        Leaing { this: T, trailing: T },
        Inner { leading: T, this: T, trailing: T },
        Trailing { leading: T, this: T },
    }

    impl<T> EstimatorImages<T> {
        pub const fn this(&self) -> &T {
            match self {
                Self::Leaing { this, .. }
                | Self::Inner { this, .. }
                | Self::Trailing { this, .. } => this,
            }
        }
    }

    impl<T: Deref> Deref for EstimatorImages<T> {
        type Target = T::Target;

        /// Equivalent to [`EstimatorImages::this`].
        fn deref(&self) -> &Self::Target {
            &*self.this()
        }
    }
}
pub use estimator_images::EstimatorImages;

/// A wrapper for implementors of the [`MinimalQuantumEstimatorSender`] trait.
pub struct MinimalQuantumEstimator<E: ?Sized>(pub(crate) E);

impl<E> MinimalQuantumEstimator<E> {
    /// Wraps the provided value with `MinimalQuantumEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

pub type GroupInTypeInImageInSystem<'a, V> = MapOutsideWhole<
    &'a AtomGroup<V>,
    MapInWhole<
        &'a AtomTypeReaderLock<V>,
        MapInWhole<&'a [AtomTypeReaderLock<V>], &'a [AtomTypeReaderLock<V>]>,
    >,
>;

/// A trait for quantum estimators that recieve
/// the calculations of quantum estimator senders
/// and outut the final value.
pub trait QuantumEstimatorReciever<T, V, Adder, Multiplier>
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

/// A trait for quantum estimators.
pub trait QuantumEstimatorSender<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad>
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
    /// and sends it to a [`QuantumEstimatorReciever`].
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
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group to the observable
    /// and sends it to a [`QuantumEstimatorReciever`].
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
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error>;
}

/// A trait for quantum estimators that do not rely on either
/// the physical nor the exchange potentials.
pub trait MinimalQuantumEstimatorSender<T, V, Adder, Multiplier>
where
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group to the observable
    /// and sends it to a [`QuantumEstimatorReciever`].
    ///
    /// Assumes this group obeys distinguishable statistics.
    fn calculate_distinguishable(
        &mut self,
        exchange_potential_is_cyclic: bool,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error>;

    /// Calculates the contribution of this group to the observable
    /// and sends it to a [`QuantumEstimatorReciever`].
    ///
    /// Assumes this group obeys bosonic statistics.
    fn calculate_bosonic(
        &mut self,
        exchange_potential_is_cyclic: bool,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error>;
}

impl<T, V, Adder, Multiplier, E> MinimalQuantumEstimatorSender<T, V, Adder, Multiplier>
    for MinimalQuantumEstimator<E>
where
    Adder: SyncAddSender<E::Output> + ?Sized,
    Multiplier: SyncMulSender<E::Output> + ?Sized,
    E: MinimalQuantumEstimatorSender<T, V, Adder, Multiplier> + ?Sized,
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
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error> {
        self.0.calculate_distinguishable(
            exchange_potential_is_cyclic,
            adder,
            multiplier,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            positions,
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
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error> {
        self.0.calculate_bosonic(
            exchange_potential_is_cyclic,
            adder,
            multiplier,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            positions,
            physical_forces,
            exchange_forces,
        )
    }
}

impl<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad, E>
    QuantumEstimatorSender<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad>
    for MinimalQuantumEstimator<E>
where
    Adder: SyncAddSender<<Self as MinimalQuantumEstimatorSender<T, V, Adder, Multiplier>>::Output>
        + ?Sized,
    Multiplier: SyncMulSender<<Self as MinimalQuantumEstimatorSender<T, V, Adder, Multiplier>>::Output>
        + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: MinimalQuantumEstimatorSender<T, V, Adder, Multiplier>,
{
    type Output = <Self as MinimalQuantumEstimatorSender<T, V, Adder, Multiplier>>::Output;
    type Error = <Self as MinimalQuantumEstimatorSender<T, V, Adder, Multiplier>>::Error;

    fn calculate_distinguishable(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Dist, &mut DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error> {
        MinimalQuantumEstimatorSender::calculate_distinguishable(
            self,
            exchange_potential.is_cyclic(),
            adder,
            multiplier,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            positions,
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
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error> {
        MinimalQuantumEstimatorSender::calculate_bosonic(
            self,
            exchange_potential.is_cyclic(),
            adder,
            multiplier,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            positions,
            physical_forces,
            exchange_forces,
        )
    }
}
