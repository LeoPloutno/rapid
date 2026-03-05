#![feature(new_range_api, ptr_metadata)]
#![allow(clippy::too_many_arguments)]
// #![warn(missing_docs)]
use std::{
    fmt::Display,
    iter,
    ops::{Add, DerefMut, Div, Mul},
    sync::{Barrier, RwLock},
    thread,
};

use arc_rw_lock::ElementRwLock;

use crate::{
    core::{AtomType, CommError, Factory, FullFactory, GroupImageHandle, GroupTypeHandle, Scheme, SchemeDependent},
    estimator::{
        classical::{
            InnerClassicalEstimator, LeadingClassicalEstimator, MainClassicalgEstimator, TrailingClassicalEstimator,
        },
        quantum::{InnerQuantumEstimator, LeadingQuantumEstimator, MainQuantumEstimator, TrailingQuantumEstimator},
    },
    output::{Estimators, ObservablesOutputOption, ValuesOutput, VectorsOutput},
    potential::{
        exchange::{
            InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential,
            quadratic::{
                InnerQuadraticExpansionExchangePotential, LeadingQuadraticExpansionExchangePotential,
                TrailingQuadraticExpansionExchangePotential,
            },
        },
        physical::PhysicalPotential,
    },
    propagator::{
        InnerPropagator, LeadingPropagator, TrailingPropagator,
        quadratic::{
            InnerQuadraticExpansionPropagator, LeadingQuadraticExpansionPropagator,
            TrailingQuadraticExpansionPropagator,
        },
    },
    stat::{Bosonic, Distinguishable, Stat},
    stride::StridesMut,
    sync_ops::{SyncAddRecv, SyncAddSend, SyncMulRecv, SyncMulSend},
    thermostat::Thermostat,
    vector::Vector,
};

pub mod core;
pub mod estimator;
pub mod marker;
pub mod output;
pub mod potential;
pub mod propagator;
pub mod stat;
mod stride;
pub mod sync_ops;
pub mod thermostat;
pub mod vector;

pub type ImageHandle<V> = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>;

fn run_step_leading_group<
    const N: usize,
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T>,
    V: Vector<N, Element = T>,
    AdderSender: SyncAddSend<T> + ?Sized,
    MultiplierSender: SyncMulSend<T> + ?Sized,
    QuantumEst: LeadingQuantumEstimator<T, V, AdderSender, MultiplierSender, Dist, DistQuad, Boson, BosonQuad, Output = Output>
        + ?Sized,
    ClassicalEst: LeadingClassicalEstimator<
            T,
            V,
            AdderSender,
            MultiplierSender,
            Dist,
            DistQuad,
            Boson,
            BosonQuad,
            Output = Output,
        > + ?Sized,
    Prop: LeadingPropagator<T, V, Phys, Dist, Boson, Therm> + ?Sized,
    PropQuad: LeadingQuadraticExpansionPropagator<T, V, Phys, DistQuad, BosonQuad, Therm> + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + Send + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    Output,
    Err: From<CommError>
        + From<AdderSender::Error>
        + From<Prop::Error>
        + From<PropQuad::Error>
        + From<QuantumEst::Error>
        + From<ClassicalEst::Error>
        + Send,
>(
    step: usize,
    barrier: &Barrier,
    shared_value: &RwLock<T>,
    atom_type: &AtomType<T>,
    group: usize,
    adder: &mut AdderSender,
    multiplier: &mut MultiplierSender,
    mut quantum_estimators: Option<&mut [&mut QuantumEst]>,
    mut classical_estimators: Option<&mut [&mut ClassicalEst]>,
    mut propagator_and_exchange_potential: Scheme<
        SchemeDependent<&mut Prop, Stat<&mut Dist, &mut Boson>>,
        SchemeDependent<&mut PropQuad, Stat<&mut DistQuad, &mut BosonQuad>>,
    >,
    physical_potential: &mut Phys,
    thermostat: &mut Therm,
    positions: &mut ImageHandle<V>,
    momenta: &mut ImageHandle<V>,
    physical_forces: &mut ImageHandle<V>,
    exchange_forces: &mut ImageHandle<V>,
) -> Result<(), Err> {
    let (group_physical_potential_energy, group_exchange_potential_energy) =
        match &mut propagator_and_exchange_potential {
            Scheme::Regular(SchemeDependent {
                propagator,
                exchange_potential,
            }) => propagator.propagate(
                step,
                physical_potential,
                exchange_potential.as_deref_mut(),
                thermostat,
                &mut *positions.write(),
                &mut *momenta.write(),
                &mut *physical_forces.write(),
                &mut *exchange_forces.write(),
            )?,
            Scheme::QuadraticExpansion(SchemeDependent {
                propagator,
                exchange_potential,
            }) => propagator.propagate(
                step,
                physical_potential,
                exchange_potential.as_deref_mut(),
                thermostat,
                &mut *positions.write(),
                &mut *momenta.write(),
                &mut *physical_forces.write(),
                &mut *exchange_forces.write(),
            )?,
        };

    let mut iter = momenta
        .read()
        .read()
        .read()
        .iter()
        .map(|momentum| T::from(0.5) * atom_type.mass.clone() * momentum.magnitude_squared());
    let tmp = iter.next().expect("`momenta` should contain at least one element");
    let group_kinetic_energy = iter.fold(tmp, |accum, elem| accum + elem);

    barrier.wait();
    adder.send(group_physical_potential_energy)?;
    barrier.wait();
    let physical_potential_energy = shared_value.read().map_err(|_| CommError::Leading { group })?.clone();

    adder.send(group_exchange_potential_energy)?;
    barrier.wait();
    let exchange_potential_energy = shared_value.read().map_err(|_| CommError::Leading { group })?.clone();

    adder.send(group_kinetic_energy)?;
    barrier.wait();
    let kinetic_energy = shared_value.read().map_err(|_| CommError::Leading { group })?.clone();

    if let Some(estimators) = quantum_estimators.as_deref_mut() {
        for estimator in estimators {
            estimator.calculate(
                adder,
                multiplier,
                match &propagator_and_exchange_potential {
                    Scheme::Regular(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::Regular(exchange_potential.as_deref())
                    }
                    Scheme::QuadraticExpansion(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::QuadraticExpansion(exchange_potential.as_deref())
                    }
                },
                physical_potential_energy.clone(),
                exchange_potential_energy.clone(),
                &positions.read_whole().map_err(|_| CommError::Leading { group })?,
                &physical_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                &exchange_forces.read_whole().map_err(|_| CommError::Leading { group })?,
            )?;
            barrier.wait();
        }
    }

    if let Some(estimators) = classical_estimators.as_deref_mut() {
        for estimator in estimators {
            estimator.calculate(
                adder,
                multiplier,
                match &propagator_and_exchange_potential {
                    Scheme::Regular(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::Regular(exchange_potential.as_deref())
                    }
                    Scheme::QuadraticExpansion(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::QuadraticExpansion(exchange_potential.as_deref())
                    }
                },
                physical_potential_energy.clone(),
                exchange_potential_energy.clone(),
                kinetic_energy.clone(),
                &positions.read_whole().map_err(|_| CommError::Leading { group })?,
                &momenta.read_whole().map_err(|_| CommError::Leading { group })?,
                &physical_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                &exchange_forces.read_whole().map_err(|_| CommError::Leading { group })?,
            )?;
            barrier.wait();
        }
    }

    Ok(())
}

fn run_step_inner_group<
    const N: usize,
    T: Clone + From<f32> + Add<Output = T> + Mul<Output = T>,
    V: Vector<N, Element = T>,
    AdderSender: SyncAddSend<T> + ?Sized,
    MultiplierSender: SyncMulSend<T> + ?Sized,
    QuantumEst: InnerQuantumEstimator<T, V, AdderSender, MultiplierSender, Dist, DistQuad, Boson, BosonQuad, Output = Output>
        + ?Sized,
    ClassicalEst: InnerClassicalEstimator<T, V, AdderSender, MultiplierSender, Dist, DistQuad, Boson, BosonQuad, Output = Output>
        + ?Sized,
    Prop: InnerPropagator<T, V, Phys, Dist, Boson, Therm> + ?Sized,
    PropQuad: InnerQuadraticExpansionPropagator<T, V, Phys, DistQuad, BosonQuad, Therm> + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + Send + ?Sized,
    Output,
    Err: From<CommError>
        + From<AdderSender::Error>
        + From<Prop::Error>
        + From<PropQuad::Error>
        + From<QuantumEst::Error>
        + From<ClassicalEst::Error>,
>(
    step: usize,
    barrier: &Barrier,
    shared_value: &RwLock<T>,
    image: usize,
    atom_type: &AtomType<T>,
    group: usize,
    adder: &mut AdderSender,
    multiplier: &mut MultiplierSender,
    mut quantum_estimators: Option<&mut [&mut QuantumEst]>,
    mut classical_estimators: Option<&mut [&mut ClassicalEst]>,
    mut propagator_and_exchange_potential: Scheme<
        SchemeDependent<&mut Prop, Stat<&mut Dist, &mut Boson>>,
        SchemeDependent<&mut PropQuad, Stat<&mut DistQuad, &mut BosonQuad>>,
    >,
    physical_potential: &mut Phys,
    thermostat: &mut Therm,
    positions: &mut ImageHandle<V>,
    momenta: &mut ImageHandle<V>,
    physical_forces: &mut ImageHandle<V>,
    exchange_forces: &mut ImageHandle<V>,
) -> Result<(), Err> {
    let (group_physical_potential_energy, group_exchange_potential_energy) =
        match &mut propagator_and_exchange_potential {
            Scheme::Regular(SchemeDependent {
                propagator,
                exchange_potential,
            }) => propagator.propagate(
                step,
                physical_potential,
                exchange_potential.as_deref_mut(),
                thermostat,
                &mut *positions.write(),
                &mut *momenta.write(),
                &mut *physical_forces.write(),
                &mut *exchange_forces.write(),
            )?,
            Scheme::QuadraticExpansion(SchemeDependent {
                propagator,
                exchange_potential,
            }) => propagator.propagate(
                step,
                physical_potential,
                exchange_potential.as_deref_mut(),
                thermostat,
                &mut *positions.write(),
                &mut *momenta.write(),
                &mut *physical_forces.write(),
                &mut *exchange_forces.write(),
            )?,
        };

    let mut iter = momenta
        .read()
        .read()
        .read()
        .iter()
        .map(|momentum| T::from(0.5) * atom_type.mass.clone() * momentum.magnitude_squared());
    let tmp = iter.next().expect("`momenta` should contain at least one element");
    let group_kinetic_energy = iter.fold(tmp, |accum, elem| accum + elem);

    barrier.wait();
    adder.send(group_physical_potential_energy)?;
    barrier.wait();
    let physical_potential_energy = shared_value
        .read()
        .map_err(|_| CommError::Inner { image, group })?
        .clone();

    adder.send(group_exchange_potential_energy)?;
    barrier.wait();
    let exchange_potential_energy = shared_value
        .read()
        .map_err(|_| CommError::Inner { image, group })?
        .clone();

    adder.send(group_kinetic_energy)?;
    barrier.wait();
    let kinetic_energy = shared_value
        .read()
        .map_err(|_| CommError::Inner { image, group })?
        .clone();

    if let Some(estimators) = quantum_estimators.as_deref_mut() {
        for estimator in estimators {
            estimator.calculate(
                adder,
                multiplier,
                match &propagator_and_exchange_potential {
                    Scheme::Regular(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::Regular(exchange_potential.as_deref())
                    }
                    Scheme::QuadraticExpansion(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::QuadraticExpansion(exchange_potential.as_deref())
                    }
                },
                physical_potential_energy.clone(),
                exchange_potential_energy.clone(),
                &positions.read_whole().map_err(|_| CommError::Inner { image, group })?,
                &physical_forces
                    .read_whole()
                    .map_err(|_| CommError::Inner { image, group })?,
                &exchange_forces
                    .read_whole()
                    .map_err(|_| CommError::Inner { image, group })?,
            )?;
            barrier.wait();
        }
    }

    if let Some(estimators) = classical_estimators.as_deref_mut() {
        for estimator in estimators {
            estimator.calculate(
                adder,
                multiplier,
                match &propagator_and_exchange_potential {
                    Scheme::Regular(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::Regular(exchange_potential.as_deref())
                    }
                    Scheme::QuadraticExpansion(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::QuadraticExpansion(exchange_potential.as_deref())
                    }
                },
                physical_potential_energy.clone(),
                exchange_potential_energy.clone(),
                kinetic_energy.clone(),
                &positions.read_whole().map_err(|_| CommError::Inner { image, group })?,
                &momenta.read_whole().map_err(|_| CommError::Inner { image, group })?,
                &physical_forces
                    .read_whole()
                    .map_err(|_| CommError::Inner { image, group })?,
                &exchange_forces
                    .read_whole()
                    .map_err(|_| CommError::Inner { image, group })?,
            )?;
            barrier.wait();
        }
    }

    Ok(())
}

fn run_step_trailing_group<
    const N: usize,
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T>,
    V: Vector<N, Element = T>,
    AdderSender: SyncAddSend<T> + ?Sized,
    MultiplierSender: SyncMulSend<T> + ?Sized,
    QuantumEst: TrailingQuantumEstimator<T, V, AdderSender, MultiplierSender, Dist, DistQuad, Boson, BosonQuad, Output = Output>
        + ?Sized,
    ClassicalEst: TrailingClassicalEstimator<
            T,
            V,
            AdderSender,
            MultiplierSender,
            Dist,
            DistQuad,
            Boson,
            BosonQuad,
            Output = Output,
        > + ?Sized,
    Prop: TrailingPropagator<T, V, Phys, Dist, Boson, Therm> + ?Sized,
    PropQuad: TrailingQuadraticExpansionPropagator<T, V, Phys, DistQuad, BosonQuad, Therm> + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    Output,
    Err: From<CommError>
        + From<AdderSender::Error>
        + From<Prop::Error>
        + From<PropQuad::Error>
        + From<QuantumEst::Error>
        + From<ClassicalEst::Error>
        + Send,
>(
    step: usize,
    barrier: &Barrier,
    shared_value: &RwLock<T>,
    atom_type: &AtomType<T>,
    group: usize,
    adder: &mut AdderSender,
    multiplier: &mut MultiplierSender,
    mut quantum_estimators: Option<&mut [&mut QuantumEst]>,
    mut classical_estimators: Option<&mut [&mut ClassicalEst]>,
    mut propagator_and_exchange_potential: Scheme<
        SchemeDependent<&mut Prop, Stat<&mut Dist, &mut Boson>>,
        SchemeDependent<&mut PropQuad, Stat<&mut DistQuad, &mut BosonQuad>>,
    >,
    physical_potential: &mut Phys,
    thermostat: &mut Therm,
    positions: &mut ImageHandle<V>,
    momenta: &mut ImageHandle<V>,
    physical_forces: &mut ImageHandle<V>,
    exchange_forces: &mut ImageHandle<V>,
) -> Result<(), Err> {
    let (group_physical_potential_energy, group_exchange_potential_energy) =
        match &mut propagator_and_exchange_potential {
            Scheme::Regular(SchemeDependent {
                propagator,
                exchange_potential,
            }) => propagator.propagate(
                step,
                physical_potential,
                exchange_potential.as_deref_mut(),
                thermostat,
                &mut *positions.write(),
                &mut *momenta.write(),
                &mut *physical_forces.write(),
                &mut *exchange_forces.write(),
            )?,
            Scheme::QuadraticExpansion(SchemeDependent {
                propagator,
                exchange_potential,
            }) => propagator.propagate(
                step,
                physical_potential,
                exchange_potential.as_deref_mut(),
                thermostat,
                &mut *positions.write(),
                &mut *momenta.write(),
                &mut *physical_forces.write(),
                &mut *exchange_forces.write(),
            )?,
        };

    let mut iter = momenta
        .read()
        .read()
        .read()
        .iter()
        .map(|momentum| T::from(0.5) * atom_type.mass.clone() * momentum.magnitude_squared());
    let tmp = iter.next().expect("`momenta` should contain at least one element");
    let group_kinetic_energy = iter.fold(tmp, |accum, elem| accum + elem);

    barrier.wait();
    adder.send(group_physical_potential_energy)?;
    barrier.wait();
    let physical_potential_energy = shared_value.read().map_err(|_| CommError::Leading { group })?.clone();

    adder.send(group_exchange_potential_energy)?;
    barrier.wait();
    let exchange_potential_energy = shared_value.read().map_err(|_| CommError::Leading { group })?.clone();

    adder.send(group_kinetic_energy)?;
    barrier.wait();
    let kinetic_energy = shared_value.read().map_err(|_| CommError::Leading { group })?.clone();

    if let Some(estimators) = quantum_estimators.as_deref_mut() {
        for estimator in estimators {
            estimator.calculate(
                adder,
                multiplier,
                match &propagator_and_exchange_potential {
                    Scheme::Regular(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::Regular(exchange_potential.as_deref())
                    }
                    Scheme::QuadraticExpansion(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::QuadraticExpansion(exchange_potential.as_deref())
                    }
                },
                physical_potential_energy.clone(),
                exchange_potential_energy.clone(),
                &positions.read_whole().map_err(|_| CommError::Leading { group })?,
                &physical_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                &exchange_forces.read_whole().map_err(|_| CommError::Leading { group })?,
            )?;
            barrier.wait();
        }
    }

    if let Some(estimators) = classical_estimators.as_deref_mut() {
        for estimator in estimators {
            estimator.calculate(
                adder,
                multiplier,
                match &propagator_and_exchange_potential {
                    Scheme::Regular(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::Regular(exchange_potential.as_deref())
                    }
                    Scheme::QuadraticExpansion(SchemeDependent { exchange_potential, .. }) => {
                        Scheme::QuadraticExpansion(exchange_potential.as_deref())
                    }
                },
                physical_potential_energy.clone(),
                exchange_potential_energy.clone(),
                kinetic_energy.clone(),
                &positions.read_whole().map_err(|_| CommError::Leading { group })?,
                &momenta.read_whole().map_err(|_| CommError::Leading { group })?,
                &physical_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                &exchange_forces.read_whole().map_err(|_| CommError::Leading { group })?,
            )?;
            barrier.wait();
        }
    }
    Ok(())
}

pub fn run<
    const N: usize,
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Display + Send + Sync,
    V: Vector<N, Element = T> + Display + Send,
    AdderReciever: SyncAddRecv<T> + ?Sized,
    AdderSender: SyncAddSend<T> + Send + ?Sized,
    MultiplierReciever: SyncMulRecv<T> + ?Sized,
    MultiplierSender: SyncMulSend<T> + Send + ?Sized,
    CoordOut: VectorsOutput<N, T, V> + ?Sized,
    QuantumEstMain: MainQuantumEstimator<T, V, AdderReciever, MultiplierReciever, Output = Output> + Send + ?Sized,
    QuantumEstLeading: LeadingQuantumEstimator<
            T,
            V,
            AdderSender,
            MultiplierSender,
            DistLeading,
            DistQuadLeading,
            BosonLeading,
            BosonQuadLeading,
            Output = Output,
        > + Send
        + ?Sized,
    QuantumEstInner: InnerQuantumEstimator<
            T,
            V,
            AdderSender,
            MultiplierSender,
            DistInner,
            DistQuadInner,
            BosonInner,
            BosonQuadInner,
            Output = Output,
        > + Send
        + ?Sized,
    QuantumEstTrailing: TrailingQuantumEstimator<
            T,
            V,
            AdderSender,
            MultiplierSender,
            DistTrailing,
            DistQuadTrailing,
            BosonTrailing,
            BosonQuadTrailing,
            Output = Output,
        > + Send
        + ?Sized,
    ClassicalEstMain: MainClassicalgEstimator<T, V, AdderReciever, MultiplierReciever, Output = Output> + ?Sized,
    ClassicalEstLeading: LeadingClassicalEstimator<
            T,
            V,
            AdderSender,
            MultiplierSender,
            DistLeading,
            DistQuadLeading,
            BosonLeading,
            BosonQuadLeading,
            Output = Output,
        > + Send
        + ?Sized,
    ClassicalEstInner: InnerClassicalEstimator<
            T,
            V,
            AdderSender,
            MultiplierSender,
            DistInner,
            DistQuadInner,
            BosonInner,
            BosonQuadInner,
            Output = Output,
        > + Send
        + ?Sized,
    ClassicalEstTrailing: TrailingClassicalEstimator<
            T,
            V,
            AdderSender,
            MultiplierSender,
            DistTrailing,
            DistQuadTrailing,
            BosonTrailing,
            BosonQuadTrailing,
            Output = Output,
        > + Send
        + ?Sized,
    EstOut: ValuesOutput<Output> + ?Sized,
    PropLeading: LeadingPropagator<T, V, Phys, DistLeading, BosonLeading, Therm> + Send + ?Sized,
    PropQuadLeading: LeadingQuadraticExpansionPropagator<T, V, Phys, DistQuadLeading, BosonQuadLeading, Therm> + Send + ?Sized,
    PropInner: InnerPropagator<T, V, Phys, DistInner, BosonInner, Therm> + Send + ?Sized,
    PropQuadInner: InnerQuadraticExpansionPropagator<T, V, Phys, DistQuadInner, BosonQuadInner, Therm> + Send + ?Sized,
    PropTrailing: TrailingPropagator<T, V, Phys, DistTrailing, BosonTrailing, Therm> + Send + ?Sized,
    PropQuadTrailing: TrailingQuadraticExpansionPropagator<T, V, Phys, DistQuadTrailing, BosonQuadTrailing, Therm> + Send + ?Sized,
    Phys: PhysicalPotential<T, V> + Send + ?Sized,
    DistLeading: LeadingExchangePotential<T, V> + Distinguishable + Send + ?Sized,
    DistQuadLeading: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + Send + ?Sized,
    DistInner: InnerExchangePotential<T, V> + Distinguishable + Send + ?Sized,
    DistQuadInner: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + Send + ?Sized,
    DistTrailing: TrailingExchangePotential<T, V> + Distinguishable + Send + ?Sized,
    DistQuadTrailing: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + Send + ?Sized,
    BosonLeading: LeadingExchangePotential<T, V> + Bosonic + Send + ?Sized,
    BosonQuadLeading: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + Send + ?Sized,
    BosonInner: InnerExchangePotential<T, V> + Bosonic + Send + ?Sized,
    BosonQuadInner: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + Send + ?Sized,
    BosonTrailing: TrailingExchangePotential<T, V> + Bosonic + Send + ?Sized,
    BosonQuadTrailing: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + Send + ?Sized,
    Therm: Thermostat<T, V> + Send + ?Sized,
    Output,
    Err: From<CommError>
        + From<AdderReciever::Error>
        + From<AdderSender::Error>
        + From<CoordOut::Error>
        + From<EstOut::Error>
        + From<PropLeading::Error>
        + From<PropQuadLeading::Error>
        + From<PropInner::Error>
        + From<PropQuadInner::Error>
        + From<PropTrailing::Error>
        + From<PropQuadTrailing::Error>
        + From<QuantumEstMain::Error>
        + From<QuantumEstLeading::Error>
        + From<QuantumEstInner::Error>
        + From<QuantumEstTrailing::Error>
        + From<ClassicalEstMain::Error>
        + From<ClassicalEstLeading::Error>
        + From<ClassicalEstInner::Error>
        + From<ClassicalEstTrailing::Error>
        + Send,
>(
    steps: usize,
    inner_images: usize,
    atom_types: &[AtomType<T>],
    groups_sizes: &[usize],
    adders: &mut (
             impl for<'a> FullFactory<
        'a,
        T,
        Main = &'a mut AdderReciever,
        Leading = &'a mut AdderSender,
        Inner = &'a mut AdderSender,
        Trailing = &'a mut AdderSender,
    > + ?Sized
         ),
    multipliers: &mut (
             impl for<'a> FullFactory<
        'a,
        T,
        Main = &'a mut MultiplierReciever,
        Leading = &'a mut MultiplierSender,
        Inner = &'a mut MultiplierSender,
        Trailing = &'a mut MultiplierSender,
    > + ?Sized
         ),
    positions_out: Option<
        &mut (impl ExactSizeIterator<Item: DerefMut<Target = CoordOut> + Send> + DoubleEndedIterator + ?Sized),
    >,
    momenta_out: Option<
        &mut (impl ExactSizeIterator<Item: DerefMut<Target = CoordOut> + Send> + DoubleEndedIterator + ?Sized),
    >,
    physical_forces_out: Option<
        &mut (impl ExactSizeIterator<Item: DerefMut<Target = CoordOut> + Send> + DoubleEndedIterator + ?Sized),
    >,
    exchange_forces_out: Option<
        &mut (impl ExactSizeIterator<Item: DerefMut<Target = CoordOut> + Send> + DoubleEndedIterator + ?Sized),
    >,
    propagators_and_exchange_potentials: Scheme<
        SchemeDependent<
            &mut (
                     impl for<'a> Factory<
                'a,
                T,
                Leading = &'a mut PropLeading,
                Inner = &'a mut PropInner,
                Trailing = &'a mut PropTrailing,
            > + ?Sized
                 ),
            &mut (
                     impl for<'a> Factory<
                'a,
                T,
                Leading = Stat<&'a mut DistLeading, &'a mut BosonLeading>,
                Inner = Stat<&'a mut DistInner, &'a mut BosonInner>,
                Trailing = Stat<&'a mut DistTrailing, &'a mut BosonTrailing>,
            > + ?Sized
                 ),
        >,
        SchemeDependent<
            &mut (
                     impl for<'a> Factory<
                'a,
                T,
                Leading = &'a mut PropQuadLeading,
                Inner = &'a mut PropQuadInner,
                Trailing = &'a mut PropQuadTrailing,
            > + ?Sized
                 ),
            &mut (
                     impl for<'a> Factory<
                'a,
                T,
                Leading = Stat<&'a mut DistQuadLeading, &'a mut BosonQuadLeading>,
                Inner = Stat<&'a mut DistQuadInner, &'a mut BosonQuadInner>,
                Trailing = Stat<&'a mut DistQuadTrailing, &'a mut BosonQuadTrailing>,
            > + ?Sized
                 ),
        >,
    >,
    estimators: ObservablesOutputOption<
        &mut [impl for<'a> FullFactory<
            'a,
            T,
            Main = &'a mut QuantumEstMain,
            Leading = &'a mut QuantumEstLeading,
            Inner = &'a mut QuantumEstInner,
            Trailing = &'a mut QuantumEstTrailing,
        >],
        &mut [impl for<'a> FullFactory<
            'a,
            T,
            Main = &'a mut ClassicalEstMain,
            Leading = &'a mut ClassicalEstLeading,
            Inner = &'a mut ClassicalEstInner,
            Trailing = &'a mut ClassicalEstTrailing,
        >],
        &mut EstOut,
    >,
    physical_potentials: &mut (
             impl for<'a> Factory<'a, T, Leading = &'a mut Phys, Inner = &'a mut Phys, Trailing = &'a mut Phys> + ?Sized
         ),
    thermostats: &mut (
             impl for<'a> Factory<'a, T, Leading = &'a mut Therm, Inner = &'a mut Therm, Trailing = &'a mut Therm> + ?Sized
         ),
    positions: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Leading = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
        Inner = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
        Trailing = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
    > + ?Sized
         ),
    momenta: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Leading = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
        Inner = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
        Trailing = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
    > + ?Sized
         ),
    physical_forces: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Leading = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
        Inner = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
        Trailing = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
    > + ?Sized
         ),
    exchange_forces: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Leading = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
        Inner = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
        Trailing = ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
    > + ?Sized
         ),
) -> Result<(), Err> {
    macro_rules! zip_items {
        ($item1:pat, $item2:pat) => {
            ($item1, $item2)
        };
        ($item:pat, $($items:pat),+) => {
            ($item, zip_items!($($items),+))
        };
    }

    macro_rules! zip_iterators {
        ($iter:expr) => {
            $iter
        };
        ($iter1:expr, $iter2:expr) => {
            $iter1.into_iter().zip($iter2)
        };
        ($iter:expr, $($iters:expr),+) => {
            $iter.zip(zip_iterators!($($iters),+))
        };
    }

    macro_rules! produce_estimators {
        ($estimators:expr) => {{
            let n_estimators = $estimators.len();
            let mut uninit_main_estimators = Box::new_uninit_slice(n_estimators);
            let mut uninit_leading_estimators = Box::new_uninit_slice(groups_sizes.len() * n_estimators);
            let mut uninit_inner_estimators = Box::new_uninit_slice(groups_sizes.len() * inner_images * n_estimators);
            let mut uninit_trailing_estimators = Box::new_uninit_slice(groups_sizes.len() * n_estimators);
            for zip_items!(
                estimator,
                uninit_main_estimator,
                uninit_leading_estimators,
                mut uninit_inner_estimators,
                uninit_trailing_estimators
            ) in zip_iterators!(
                $estimators.iter_mut(),
                uninit_main_estimators.iter_mut(),
                StridesMut::from_slice(&mut uninit_leading_estimators, n_estimators),
                StridesMut::from_slice(&mut uninit_inner_estimators, n_estimators),
                StridesMut::from_slice(&mut uninit_trailing_estimators, n_estimators)
            ) {
                let (main_estimator, leading_estimators, inner_estimators_iter, trailing_estimators) =
                    estimator.produce(inner_images, atom_types, groups_sizes);

                assert_eq!(leading_estimators.len(), groups_sizes.len());
                assert_eq!(inner_estimators_iter.len(), inner_images);
                assert_eq!(trailing_estimators.len(), groups_sizes.len());

                uninit_main_estimator.write(main_estimator);

                for zip_items!(
                    uninit_leading_estimator,
                    uninit_trailing_estimator,
                    leading_estimator,
                    trailing_estimator
                ) in zip_iterators!(
                    uninit_leading_estimators,
                    uninit_trailing_estimators,
                    leading_estimators,
                    trailing_estimators
                ) {
                    uninit_leading_estimator.write(leading_estimator);
                    uninit_trailing_estimator.write(trailing_estimator);
                }

                for inner_estimators in inner_estimators_iter {
                    assert_eq!(inner_estimators.len(), groups_sizes.len());

                    for (uninit_inner_estimator, inner_estimator) in uninit_inner_estimators
                        .by_ref()
                        .take(groups_sizes.len())
                        .zip(inner_estimators)
                    {
                        uninit_inner_estimator.write(inner_estimator);
                    }
                }
            }
            // SAFETY: Initialized all elements above.
            unsafe {
                (
                    n_estimators,
                    uninit_main_estimators.assume_init(),
                    uninit_leading_estimators.assume_init(),
                    uninit_inner_estimators.assume_init(),
                    uninit_trailing_estimators.assume_init(),
                )
            }
        }};
    }

    let (mut leading_positions_out, mut inner_positions_out_iter, mut trailing_positions_out) =
        if let Some(iter) = positions_out {
            assert_eq!(iter.len(), inner_images + 2);

            let leading = iter
                .next()
                .expect("`positions_out` yields at least `inner_images + 2` elements");
            let trailing = iter
                .next_back()
                .expect("`positions_out` yields at least `inner_images + 2` elements");
            (Some(leading), Some(iter), Some(trailing))
        } else {
            (None, None, None)
        };
    let (mut leading_momenta_out, mut inner_momenta_out_iter, mut trailing_momenta_out) =
        if let Some(iter) = momenta_out {
            assert_eq!(iter.len(), inner_images + 2);

            let leading = iter
                .next()
                .expect("`positions_out` yields at least `inner_images + 2` elements");
            let trailing = iter
                .next_back()
                .expect("`positions_out` yields at least `inner_images + 2` elements");
            (Some(leading), Some(iter), Some(trailing))
        } else {
            (None, None, None)
        };
    let (mut leading_physical_forces_out, mut inner_physical_forces_out_iter, mut trailing_physical_forces_out) =
        if let Some(iter) = physical_forces_out {
            assert_eq!(iter.len(), inner_images + 2);

            let leading = iter
                .next()
                .expect("`positions_out` yields at least `inner_images + 2` elements");
            let trailing = iter
                .next_back()
                .expect("`positions_out` yields at least `inner_images + 2` elements");
            (Some(leading), Some(iter), Some(trailing))
        } else {
            (None, None, None)
        };
    let (mut leading_exchange_forces_out, mut inner_exchange_forces_out_iter, mut trailing_exchange_forces_out) =
        if let Some(iter) = exchange_forces_out {
            assert_eq!(iter.len(), inner_images + 2);

            let leading = iter
                .next()
                .expect("`positions_out` yields at least `inner_images + 2` elements");
            let trailing = iter
                .next_back()
                .expect("`positions_out` yields at least `inner_images + 2` elements");
            (Some(leading), Some(iter), Some(trailing))
        } else {
            (None, None, None)
        };

    let (
        mut main_estimators,
        (
            n_quantum_estimators,
            mut leading_quantum_estimators,
            mut inner_quantum_estimators,
            mut trailing_quantum_estimators,
        ),
        (n_debug_estimators, mut leading_debug_estimators, mut inner_debug_estimators, mut trailing_debug_estimators),
    ) = match estimators {
        ObservablesOutputOption::None => (
            ObservablesOutputOption::None,
            (0, None, None, None),
            (0, None, None, None),
        ),
        ObservablesOutputOption::Quantum(Estimators { estimators, stream }) => {
            let (n_estimators, main_estimators, leading_estimators, inner_estimators, trailing_estimators) =
                produce_estimators!(estimators);
            (
                ObservablesOutputOption::Quantum(Estimators {
                    estimators: main_estimators,
                    stream,
                }),
                (
                    n_estimators,
                    Some(leading_estimators),
                    Some(inner_estimators),
                    Some(trailing_estimators),
                ),
                (0, None, None, None),
            )
        }
        ObservablesOutputOption::Classical(Estimators { estimators, stream }) => {
            let (n_estimators, main_estimators, leading_estimators, inner_estimators, trailing_estimators) =
                produce_estimators!(estimators);
            (
                ObservablesOutputOption::Classical(Estimators {
                    estimators: main_estimators,
                    stream,
                }),
                (0, None, None, None),
                (
                    n_estimators,
                    Some(leading_estimators),
                    Some(inner_estimators),
                    Some(trailing_estimators),
                ),
            )
        }
        ObservablesOutputOption::Shared {
            quantum_estimators,
            debug_estimators,
            stream,
        } => {
            let (
                n_quantum_estimators,
                main_quantum_estimators,
                leading_quantum_estimators,
                inner_quantum_estimators,
                trailing_quantum_estimators,
            ) = produce_estimators!(quantum_estimators);
            let (
                n_debug_estimators,
                main_debug_estimators,
                leading_debug_estimators,
                inner_debug_estimators,
                trailing_debug_estimators,
            ) = produce_estimators!(debug_estimators);
            (
                ObservablesOutputOption::Shared {
                    quantum_estimators: main_quantum_estimators,
                    debug_estimators: main_debug_estimators,
                    stream,
                },
                (
                    n_quantum_estimators,
                    Some(leading_quantum_estimators),
                    Some(inner_quantum_estimators),
                    Some(trailing_quantum_estimators),
                ),
                (
                    n_debug_estimators,
                    Some(leading_debug_estimators),
                    Some(inner_debug_estimators),
                    Some(trailing_debug_estimators),
                ),
            )
        }
        ObservablesOutputOption::Separate { quantum, debug } => {
            let (
                n_quantum_estimators,
                main_quantum_estimators,
                leading_quantum_estimators,
                inner_quantum_estimators,
                trailing_quantum_estimators,
            ) = produce_estimators!(quantum.estimators);
            let (
                n_debug_estimators,
                main_debug_estimators,
                leading_debug_estimators,
                inner_debug_estimators,
                trailing_debug_estimators,
            ) = produce_estimators!(debug.estimators);
            (
                ObservablesOutputOption::Separate {
                    quantum: Estimators {
                        estimators: main_quantum_estimators,
                        stream: quantum.stream,
                    },
                    debug: Estimators {
                        estimators: main_debug_estimators,
                        stream: debug.stream,
                    },
                },
                (
                    n_quantum_estimators,
                    Some(leading_quantum_estimators),
                    Some(inner_quantum_estimators),
                    Some(trailing_quantum_estimators),
                ),
                (
                    n_debug_estimators,
                    Some(leading_debug_estimators),
                    Some(inner_debug_estimators),
                    Some(trailing_debug_estimators),
                ),
            )
        }
    };

    let barrier = Barrier::new(inner_images + 3);
    let shared_value = RwLock::new(T::default());

    let barrier = &barrier;
    let shared_value = &shared_value;

    let (main_adder, leading_adders, inner_adders_iter, trailing_adders) =
        adders.produce(inner_images, atom_types, groups_sizes);
    assert_eq!(leading_adders.len(), groups_sizes.len());
    assert_eq!(inner_adders_iter.len(), inner_images);
    assert_eq!(trailing_adders.len(), groups_sizes.len());

    let (main_multiplier, leading_multipliers, inner_multipliers_iter, trailing_multipliers) =
        multipliers.produce(inner_images, atom_types, groups_sizes);
    assert_eq!(leading_multipliers.len(), groups_sizes.len());
    assert_eq!(inner_multipliers_iter.len(), inner_images);
    assert_eq!(trailing_multipliers.len(), groups_sizes.len());

    let (
        mut leading_propagators_and_exchange_potentials,
        mut inner_propagators_and_exchange_potentials_iter,
        mut trailing_propagators_and_exchange_potentials,
    ) = match propagators_and_exchange_potentials {
        Scheme::Regular(SchemeDependent {
            propagator,
            exchange_potential,
        }) => {
            let (leading_propagators, inner_propagators_iter, trailing_propagators) =
                propagator.produce(inner_images, atom_types, groups_sizes);
            assert_eq!(leading_propagators.len(), groups_sizes.len());
            assert_eq!(inner_propagators_iter.len(), inner_images);
            assert_eq!(trailing_propagators.len(), groups_sizes.len());

            let (leading_exchange_potentials, inner_exchange_potentials_iter, trailing_exchange_potentials) =
                exchange_potential.produce(inner_images, atom_types, groups_sizes);
            assert_eq!(leading_exchange_potentials.len(), groups_sizes.len());
            assert_eq!(inner_exchange_potentials_iter.len(), inner_images);
            assert_eq!(trailing_exchange_potentials.len(), groups_sizes.len());

            (
                Scheme::Regular(SchemeDependent {
                    propagator: leading_propagators,
                    exchange_potential: leading_exchange_potentials,
                }),
                Scheme::Regular(SchemeDependent {
                    propagator: inner_propagators_iter,
                    exchange_potential: inner_exchange_potentials_iter,
                }),
                Scheme::Regular(SchemeDependent {
                    propagator: trailing_propagators,
                    exchange_potential: trailing_exchange_potentials,
                }),
            )
        }
        Scheme::QuadraticExpansion(SchemeDependent {
            propagator,
            exchange_potential,
        }) => {
            let (leading_propagators, inner_propagators_iter, trailing_propagators) =
                propagator.produce(inner_images, atom_types, groups_sizes);
            assert_eq!(leading_propagators.len(), groups_sizes.len());
            assert_eq!(inner_propagators_iter.len(), inner_images);
            assert_eq!(trailing_propagators.len(), groups_sizes.len());

            let (leading_exchange_potentials, inner_exchange_potentials_iter, trailing_exchange_potentials) =
                exchange_potential.produce(inner_images, atom_types, groups_sizes);
            assert_eq!(leading_exchange_potentials.len(), groups_sizes.len());
            assert_eq!(inner_exchange_potentials_iter.len(), inner_images);
            assert_eq!(trailing_exchange_potentials.len(), groups_sizes.len());

            (
                Scheme::QuadraticExpansion(SchemeDependent {
                    propagator: leading_propagators,
                    exchange_potential: leading_exchange_potentials,
                }),
                Scheme::QuadraticExpansion(SchemeDependent {
                    propagator: inner_propagators_iter,
                    exchange_potential: inner_exchange_potentials_iter,
                }),
                Scheme::QuadraticExpansion(SchemeDependent {
                    propagator: trailing_propagators,
                    exchange_potential: trailing_exchange_potentials,
                }),
            )
        }
    };
    let (leading_physical_potentials, inner_physical_potentials_iter, trailing_physical_potentials) =
        physical_potentials.produce(inner_images, atom_types, groups_sizes);
    assert_eq!(leading_physical_potentials.len(), groups_sizes.len());
    assert_eq!(inner_physical_potentials_iter.len(), inner_images);
    assert_eq!(trailing_physical_potentials.len(), groups_sizes.len());

    let (leading_thermostats, inner_thermostats_iter, trailing_thermostats) =
        thermostats.produce(inner_images, atom_types, groups_sizes);
    assert_eq!(leading_thermostats.len(), groups_sizes.len());
    assert_eq!(inner_thermostats_iter.len(), inner_images);
    assert_eq!(trailing_thermostats.len(), groups_sizes.len());

    let (leading_positions, inner_positions_iter, trailing_positions) =
        positions.produce(inner_images, atom_types, groups_sizes);
    assert_eq!(leading_positions.len(), groups_sizes.len());
    assert_eq!(inner_positions_iter.len(), inner_images);
    assert_eq!(trailing_positions.len(), groups_sizes.len());

    let (leading_momenta, inner_momenta_iter, trailing_momenta) =
        momenta.produce(inner_images, atom_types, groups_sizes);
    assert_eq!(leading_momenta.len(), groups_sizes.len());
    assert_eq!(inner_momenta_iter.len(), inner_images);
    assert_eq!(trailing_momenta.len(), groups_sizes.len());

    let (leading_physical_forces, inner_physical_forces_iter, trailing_physical_forces) =
        physical_forces.produce(inner_images, atom_types, groups_sizes);
    assert_eq!(leading_physical_forces.len(), groups_sizes.len());
    assert_eq!(inner_physical_forces_iter.len(), inner_images);
    assert_eq!(trailing_physical_forces.len(), groups_sizes.len());

    let (leading_exchange_forces, inner_exchange_forces_iter, trailing_exchange_forces) =
        exchange_forces.produce(inner_images, atom_types, groups_sizes);
    assert_eq!(leading_exchange_forces.len(), groups_sizes.len());
    assert_eq!(inner_exchange_forces_iter.len(), inner_images);
    assert_eq!(trailing_exchange_forces.len(), groups_sizes.len());

    let index_smallest_group = groups_sizes
        .iter()
        .enumerate()
        .min_by(|(_, element_0), (_, element_1)| element_0.cmp(element_1))
        .expect("`groups_sizes` should contain at least one element")
        .0;

    thread::scope(|s| {
        // Leading image.
        let mut atom_types_iter = atom_types.iter();
        let mut leading_quantum_estimators_iter = leading_quantum_estimators
            .as_deref_mut()
            .map(|estimators| estimators.chunks_exact_mut(n_quantum_estimators));
        let mut leading_debug_estimators_iter = leading_debug_estimators
            .as_deref_mut()
            .map(|estimators| estimators.chunks_exact_mut(n_debug_estimators));
        let mut leading_iter = zip_iterators!(
            iter::successors(
                Some((atom_types.first().expect("`types` should contain at least one type"), 0)),
                |&(atom_type, group)| {
                    let group = group + 1;
                    if atom_type.groups.contains(&group) {
                        Some((atom_type, group))
                    } else {
                        atom_types_iter.next().map(|atom_type| (atom_type, group))
                    }
                }
            ),
            leading_adders,
            leading_multipliers,
            iter::from_fn(|| {
                match &mut leading_quantum_estimators_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut leading_debug_estimators_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut leading_propagators_and_exchange_potentials {
                    Scheme::Regular(SchemeDependent {
                        propagator,
                        exchange_potential,
                    }) => match (propagator.next(), exchange_potential.next()) {
                        (Some(propagator), Some(exchange_potential)) => Some(Scheme::Regular(SchemeDependent {
                            propagator,
                            exchange_potential,
                        })),
                        _ => None,
                    },
                    Scheme::QuadraticExpansion(SchemeDependent {
                        propagator,
                        exchange_potential,
                    }) => match (propagator.next(), exchange_potential.next()) {
                        (Some(propagator), Some(exchange_potential)) => {
                            Some(Scheme::QuadraticExpansion(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }))
                        }
                        _ => None,
                    },
                }
            }),
            leading_physical_potentials,
            leading_thermostats,
            leading_positions,
            leading_momenta,
            leading_physical_forces,
            leading_exchange_forces
        );

        for zip_items!(
            (atom_type, group),
            adder,
            multiplier,
            mut quantum_estimators,
            mut debug_estimators,
            mut propagator_and_exchange_potential,
            physical_potential,
            thermostat,
            mut positions,
            mut momenta,
            mut physical_forces,
            mut exchange_forces
        ) in leading_iter.by_ref().take(index_smallest_group)
        {
            s.spawn::<_, Result<_, Err>>(move || {
                for step in 0..steps {
                    let step_result: Result<_, Err> = run_step_leading_group(
                        step,
                        barrier,
                        shared_value,
                        atom_type,
                        group,
                        adder,
                        multiplier,
                        quantum_estimators.as_deref_mut(),
                        debug_estimators.as_deref_mut(),
                        match &mut propagator_and_exchange_potential {
                            Scheme::Regular(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::Regular(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                            Scheme::QuadraticExpansion(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::QuadraticExpansion(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                        },
                        physical_potential,
                        thermostat,
                        &mut positions,
                        &mut momenta,
                        &mut physical_forces,
                        &mut exchange_forces,
                    );
                    step_result?;

                    barrier.wait();
                }
                Ok(())
            });
        }

        {
            let zip_items!(
                (atom_type, group),
                adder,
                multiplier,
                mut quantum_estimators,
                mut debug_estimators,
                mut propagator_and_exchange_potential,
                physical_potential,
                thermostat,
                mut positions,
                mut momenta,
                mut physical_forces,
                mut exchange_forces
            ) = leading_iter
                .next()
                .expect("The number of groups is greater than the index of the smalles one");
            s.spawn::<_, Result<_, Err>>(move || {
                for step in 0..steps {
                    let step_result: Result<_, Err> = run_step_leading_group(
                        step,
                        barrier,
                        shared_value,
                        atom_type,
                        group,
                        adder,
                        multiplier,
                        quantum_estimators.as_deref_mut(),
                        debug_estimators.as_deref_mut(),
                        match &mut propagator_and_exchange_potential {
                            Scheme::Regular(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::Regular(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                            Scheme::QuadraticExpansion(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::QuadraticExpansion(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                        },
                        physical_potential,
                        thermostat,
                        &mut positions,
                        &mut momenta,
                        &mut physical_forces,
                        &mut exchange_forces,
                    );
                    step_result?;

                    if let Some(positions_out) = leading_positions_out.as_deref_mut() {
                        positions_out.write(
                            step,
                            &positions
                                .read()
                                .read_image()
                                .map_err(|_| CommError::Leading { group })?,
                        )?;
                    }
                    if let Some(momenta_out) = leading_momenta_out.as_deref_mut() {
                        momenta_out.write(
                            step,
                            &momenta.read().read_image().map_err(|_| CommError::Leading { group })?,
                        )?;
                    }
                    if let Some(physical_forces_out) = leading_physical_forces_out.as_deref_mut() {
                        physical_forces_out.write(
                            step,
                            &physical_forces
                                .read()
                                .read_image()
                                .map_err(|_| CommError::Leading { group })?,
                        )?;
                    }
                    if let Some(exchange_forces_out) = leading_exchange_forces_out.as_deref_mut() {
                        exchange_forces_out.write(
                            step,
                            &exchange_forces
                                .read()
                                .read_image()
                                .map_err(|_| CommError::Leading { group })?,
                        )?;
                    }

                    barrier.wait();
                }
                Ok(())
            });
        }

        for zip_items!(
            (atom_type, group),
            adder,
            multiplier,
            mut quantum_estimators,
            mut debug_estimators,
            mut propagator_and_exchange_potential,
            physical_potential,
            thermostat,
            mut positions,
            mut momenta,
            mut physical_forces,
            mut exchange_forces
        ) in leading_iter
        {
            s.spawn::<_, Result<_, Err>>(move || {
                for step in 0..steps {
                    let step_result: Result<_, Err> = run_step_leading_group(
                        step,
                        barrier,
                        shared_value,
                        atom_type,
                        group,
                        adder,
                        multiplier,
                        quantum_estimators.as_deref_mut(),
                        debug_estimators.as_deref_mut(),
                        match &mut propagator_and_exchange_potential {
                            Scheme::Regular(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::Regular(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                            Scheme::QuadraticExpansion(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::QuadraticExpansion(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                        },
                        physical_potential,
                        thermostat,
                        &mut positions,
                        &mut momenta,
                        &mut physical_forces,
                        &mut exchange_forces,
                    );
                    step_result?;

                    barrier.wait();
                }

                Ok(())
            });
        }

        // Inner images.
        let mut inner_quantum_estimators_iter = inner_quantum_estimators
            .as_deref_mut()
            .map(|estimators| estimators.chunks_exact_mut(groups_sizes.len() * n_quantum_estimators));
        let mut inner_debug_estimators_iter = inner_debug_estimators
            .as_deref_mut()
            .map(|estimators| estimators.chunks_exact_mut(groups_sizes.len() * n_debug_estimators));
        for zip_items!(
            image,
            inner_adders,
            inner_multipliers,
            mut inner_positions_out,
            mut inner_momenta_out,
            mut inner_physical_forces_out,
            mut inner_exchange_forces_out,
            inner_quantum_estimators,
            inner_debug_estimators,
            mut inner_propagators_and_exchange_potentials,
            inner_physical_potentials,
            inner_thermostats,
            inner_positions,
            inner_momenta,
            inner_physical_forces,
            inner_exchange_forces
        ) in zip_iterators!(
            1..=inner_images,
            inner_adders_iter,
            inner_multipliers_iter,
            iter::from_fn(|| {
                match &mut inner_positions_out_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut inner_momenta_out_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut inner_physical_forces_out_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut inner_exchange_forces_out_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut inner_quantum_estimators_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut inner_debug_estimators_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut inner_propagators_and_exchange_potentials_iter {
                    Scheme::Regular(SchemeDependent {
                        propagator,
                        exchange_potential,
                    }) => match (propagator.next(), exchange_potential.next()) {
                        (Some(propagator), Some(exchange_potential)) => Some(Scheme::Regular(SchemeDependent {
                            propagator,
                            exchange_potential,
                        })),
                        _ => None,
                    },
                    Scheme::QuadraticExpansion(SchemeDependent {
                        propagator,
                        exchange_potential,
                    }) => match (propagator.next(), exchange_potential.next()) {
                        (Some(propagator), Some(exchange_potential)) => {
                            Some(Scheme::QuadraticExpansion(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }))
                        }
                        _ => None,
                    },
                }
            }),
            inner_physical_potentials_iter,
            inner_thermostats_iter,
            inner_positions_iter,
            inner_momenta_iter,
            inner_physical_forces_iter,
            inner_exchange_forces_iter
        ) {
            let mut atom_types_iter = atom_types.iter();
            let mut quantum_estimators_iter =
                inner_quantum_estimators.map(|estimators| estimators.chunks_exact_mut(n_quantum_estimators));
            let mut debug_estimators_iter =
                inner_debug_estimators.map(|estimators| estimators.chunks_exact_mut(n_debug_estimators));
            let mut inner_iter = zip_iterators!(
                iter::successors(
                    Some((atom_types.first().expect("`types` should contain at least one type"), 0)),
                    |&(atom_type, group)| {
                        let group = group + 1;
                        if atom_type.groups.contains(&group) {
                            Some((atom_type, group))
                        } else {
                            atom_types_iter.next().map(|atom_type| (atom_type, group))
                        }
                    }
                ),
                inner_adders,
                inner_multipliers,
                iter::from_fn(|| {
                    match &mut quantum_estimators_iter {
                        Some(iter) => iter.next().map(Some),
                        None => Some(None),
                    }
                }),
                iter::from_fn(|| {
                    match &mut debug_estimators_iter {
                        Some(iter) => iter.next().map(Some),
                        None => Some(None),
                    }
                }),
                iter::from_fn(|| {
                    match &mut inner_propagators_and_exchange_potentials {
                        Scheme::Regular(SchemeDependent {
                            propagator,
                            exchange_potential,
                        }) => match (propagator.next(), exchange_potential.next()) {
                            (Some(propagator), Some(exchange_potential)) => Some(Scheme::Regular(SchemeDependent {
                                propagator,
                                exchange_potential,
                            })),
                            _ => None,
                        },
                        Scheme::QuadraticExpansion(SchemeDependent {
                            propagator,
                            exchange_potential,
                        }) => match (propagator.next(), exchange_potential.next()) {
                            (Some(propagator), Some(exchange_potential)) => {
                                Some(Scheme::QuadraticExpansion(SchemeDependent {
                                    propagator,
                                    exchange_potential,
                                }))
                            }
                            _ => None,
                        },
                    }
                }),
                inner_physical_potentials,
                inner_thermostats,
                inner_positions,
                inner_momenta,
                inner_physical_forces,
                inner_exchange_forces
            );

            for zip_items!(
                (atom_type, group),
                adder,
                multiplier,
                mut quantum_estimators,
                mut debug_estimators,
                mut propagator_and_exchange_potential,
                physical_potential,
                thermostat,
                mut positions,
                mut momenta,
                mut physical_forces,
                mut exchange_forces
            ) in inner_iter.by_ref().take(index_smallest_group)
            {
                s.spawn::<_, Result<_, Err>>(move || {
                    for step in 0..steps {
                        let step_result: Result<_, Err> = run_step_inner_group(
                            step,
                            barrier,
                            shared_value,
                            image,
                            atom_type,
                            group,
                            adder,
                            multiplier,
                            quantum_estimators.as_deref_mut(),
                            debug_estimators.as_deref_mut(),
                            match &mut propagator_and_exchange_potential {
                                Scheme::Regular(SchemeDependent {
                                    propagator,
                                    exchange_potential,
                                }) => Scheme::Regular(SchemeDependent {
                                    propagator: *propagator,
                                    exchange_potential: exchange_potential.as_deref_mut(),
                                }),
                                Scheme::QuadraticExpansion(SchemeDependent {
                                    propagator,
                                    exchange_potential,
                                }) => Scheme::QuadraticExpansion(SchemeDependent {
                                    propagator: *propagator,
                                    exchange_potential: exchange_potential.as_deref_mut(),
                                }),
                            },
                            physical_potential,
                            thermostat,
                            &mut positions,
                            &mut momenta,
                            &mut physical_forces,
                            &mut exchange_forces,
                        );
                        step_result?;

                        barrier.wait();
                    }
                    Ok(())
                });
            }

            {
                let zip_items!(
                    (atom_type, group),
                    adder,
                    multiplier,
                    mut quantum_estimators,
                    mut debug_estimators,
                    mut propagator_and_exchange_potential,
                    physical_potential,
                    thermostat,
                    mut positions,
                    mut momenta,
                    mut physical_forces,
                    mut exchange_forces
                ) = inner_iter
                    .next()
                    .expect("The number of groups is greater than the index of the smalles one");

                s.spawn::<_, Result<_, Err>>(move || {
                    for step in 0..steps {
                        let step_result: Result<_, Err> = run_step_inner_group(
                            step,
                            barrier,
                            shared_value,
                            image,
                            atom_type,
                            group,
                            adder,
                            multiplier,
                            quantum_estimators.as_deref_mut(),
                            debug_estimators.as_deref_mut(),
                            match &mut propagator_and_exchange_potential {
                                Scheme::Regular(SchemeDependent {
                                    propagator,
                                    exchange_potential,
                                }) => Scheme::Regular(SchemeDependent {
                                    propagator: *propagator,
                                    exchange_potential: exchange_potential.as_deref_mut(),
                                }),
                                Scheme::QuadraticExpansion(SchemeDependent {
                                    propagator,
                                    exchange_potential,
                                }) => Scheme::QuadraticExpansion(SchemeDependent {
                                    propagator: *propagator,
                                    exchange_potential: exchange_potential.as_deref_mut(),
                                }),
                            },
                            physical_potential,
                            thermostat,
                            &mut positions,
                            &mut momenta,
                            &mut physical_forces,
                            &mut exchange_forces,
                        );
                        step_result?;

                        if let Some(positions_out) = inner_positions_out.as_deref_mut() {
                            positions_out.write(
                                step,
                                &positions
                                    .read()
                                    .read_image()
                                    .map_err(|_| CommError::Leading { group })?,
                            )?;
                        }
                        if let Some(momenta_out) = inner_momenta_out.as_deref_mut() {
                            momenta_out.write(
                                step,
                                &momenta.read().read_image().map_err(|_| CommError::Leading { group })?,
                            )?;
                        }
                        if let Some(physical_forces_out) = inner_physical_forces_out.as_deref_mut() {
                            physical_forces_out.write(
                                step,
                                &physical_forces
                                    .read()
                                    .read_image()
                                    .map_err(|_| CommError::Leading { group })?,
                            )?;
                        }
                        if let Some(exchange_forces_out) = inner_exchange_forces_out.as_deref_mut() {
                            exchange_forces_out.write(
                                step,
                                &exchange_forces
                                    .read()
                                    .read_image()
                                    .map_err(|_| CommError::Leading { group })?,
                            )?;
                        }

                        barrier.wait();
                    }
                    Ok(())
                });
            }

            for zip_items!(
                (atom_type, group),
                adder,
                multiplier,
                mut quantum_estimators,
                mut debug_estimators,
                mut propagator_and_exchange_potential,
                physical_potential,
                thermostat,
                mut positions,
                mut momenta,
                mut physical_forces,
                mut exchange_forces
            ) in inner_iter
            {
                s.spawn::<_, Result<_, Err>>(move || {
                    for step in 0..steps {
                        let step_result: Result<_, Err> = run_step_inner_group(
                            step,
                            barrier,
                            shared_value,
                            image,
                            atom_type,
                            group,
                            adder,
                            multiplier,
                            quantum_estimators.as_deref_mut(),
                            debug_estimators.as_deref_mut(),
                            match &mut propagator_and_exchange_potential {
                                Scheme::Regular(SchemeDependent {
                                    propagator,
                                    exchange_potential,
                                }) => Scheme::Regular(SchemeDependent {
                                    propagator: *propagator,
                                    exchange_potential: exchange_potential.as_deref_mut(),
                                }),
                                Scheme::QuadraticExpansion(SchemeDependent {
                                    propagator,
                                    exchange_potential,
                                }) => Scheme::QuadraticExpansion(SchemeDependent {
                                    propagator: *propagator,
                                    exchange_potential: exchange_potential.as_deref_mut(),
                                }),
                            },
                            physical_potential,
                            thermostat,
                            &mut positions,
                            &mut momenta,
                            &mut physical_forces,
                            &mut exchange_forces,
                        );
                        step_result?;

                        barrier.wait();
                    }
                    Ok(())
                });
            }
        }

        // Trailing image.
        let mut atom_types_iter = atom_types.iter();
        let mut trailing_quantum_estimators_iter = trailing_quantum_estimators
            .as_deref_mut()
            .map(|estimators| estimators.chunks_exact_mut(n_quantum_estimators));
        let mut trailing_debug_estimators_iter = trailing_debug_estimators
            .as_deref_mut()
            .map(|estimators| estimators.chunks_exact_mut(n_debug_estimators));
        let mut trailing_iter = zip_iterators!(
            iter::successors(
                Some((atom_types.first().expect("`types` should contain at least one type"), 0)),
                |&(atom_type, group)| {
                    let group = group + 1;
                    if atom_type.groups.contains(&group) {
                        Some((atom_type, group))
                    } else {
                        atom_types_iter.next().map(|atom_type| (atom_type, group))
                    }
                }
            ),
            trailing_adders,
            trailing_multipliers,
            iter::from_fn(|| {
                match &mut trailing_quantum_estimators_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut trailing_debug_estimators_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut trailing_propagators_and_exchange_potentials {
                    Scheme::Regular(SchemeDependent {
                        propagator,
                        exchange_potential,
                    }) => match (propagator.next(), exchange_potential.next()) {
                        (Some(propagator), Some(exchange_potential)) => Some(Scheme::Regular(SchemeDependent {
                            propagator,
                            exchange_potential,
                        })),
                        _ => None,
                    },
                    Scheme::QuadraticExpansion(SchemeDependent {
                        propagator,
                        exchange_potential,
                    }) => match (propagator.next(), exchange_potential.next()) {
                        (Some(propagator), Some(exchange_potential)) => {
                            Some(Scheme::QuadraticExpansion(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }))
                        }
                        _ => None,
                    },
                }
            }),
            trailing_physical_potentials,
            trailing_thermostats,
            trailing_positions,
            trailing_momenta,
            trailing_physical_forces,
            trailing_exchange_forces
        );

        for zip_items!(
            (atom_type, group),
            adder,
            multiplier,
            mut quantum_estimators,
            mut debug_estimators,
            mut propagator_and_exchange_potential,
            physical_potential,
            thermostat,
            mut positions,
            mut momenta,
            mut physical_forces,
            mut exchange_forces
        ) in trailing_iter.by_ref().take(index_smallest_group)
        {
            s.spawn::<_, Result<_, Err>>(move || {
                for step in 0..steps {
                    let step_result: Result<_, Err> = run_step_trailing_group(
                        step,
                        barrier,
                        shared_value,
                        atom_type,
                        group,
                        adder,
                        multiplier,
                        quantum_estimators.as_deref_mut(),
                        debug_estimators.as_deref_mut(),
                        match &mut propagator_and_exchange_potential {
                            Scheme::Regular(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::Regular(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                            Scheme::QuadraticExpansion(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::QuadraticExpansion(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                        },
                        physical_potential,
                        thermostat,
                        &mut positions,
                        &mut momenta,
                        &mut physical_forces,
                        &mut exchange_forces,
                    );
                    step_result?;

                    barrier.wait();
                }

                Ok(())
            });
        }

        {
            let zip_items!(
                (atom_type, group),
                adder,
                multiplier,
                mut quantum_estimators,
                mut debug_estimators,
                mut propagator_and_exchange_potential,
                physical_potential,
                thermostat,
                mut positions,
                mut momenta,
                mut physical_forces,
                mut exchange_forces
            ) = trailing_iter
                .next()
                .expect("The number of groups is greater than the index of the smalles one");

            s.spawn::<_, Result<_, Err>>(move || {
                for step in 0..steps {
                    let step_result: Result<_, Err> = run_step_trailing_group(
                        step,
                        barrier,
                        shared_value,
                        atom_type,
                        group,
                        adder,
                        multiplier,
                        quantum_estimators.as_deref_mut(),
                        debug_estimators.as_deref_mut(),
                        match &mut propagator_and_exchange_potential {
                            Scheme::Regular(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::Regular(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                            Scheme::QuadraticExpansion(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::QuadraticExpansion(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                        },
                        physical_potential,
                        thermostat,
                        &mut positions,
                        &mut momenta,
                        &mut physical_forces,
                        &mut exchange_forces,
                    );
                    step_result?;

                    if let Some(positions_out) = trailing_positions_out.as_deref_mut() {
                        positions_out.write(
                            step,
                            &positions
                                .read()
                                .read_image()
                                .map_err(|_| CommError::Leading { group })?,
                        )?;
                    }
                    if let Some(momenta_out) = trailing_momenta_out.as_deref_mut() {
                        momenta_out.write(
                            step,
                            &momenta.read().read_image().map_err(|_| CommError::Leading { group })?,
                        )?;
                    }
                    if let Some(physical_forces_out) = trailing_physical_forces_out.as_deref_mut() {
                        physical_forces_out.write(
                            step,
                            &physical_forces
                                .read()
                                .read_image()
                                .map_err(|_| CommError::Leading { group })?,
                        )?;
                    }
                    if let Some(exchange_forces_out) = trailing_exchange_forces_out.as_deref_mut() {
                        exchange_forces_out.write(
                            step,
                            &exchange_forces
                                .read()
                                .read_image()
                                .map_err(|_| CommError::Leading { group })?,
                        )?;
                    }

                    barrier.wait();
                }

                Ok(())
            });
        }

        for zip_items!(
            (atom_type, group),
            adder,
            multiplier,
            mut quantum_estimators,
            mut debug_estimators,
            mut propagator_and_exchange_potential,
            physical_potential,
            thermostat,
            mut positions,
            mut momenta,
            mut physical_forces,
            mut exchange_forces
        ) in trailing_iter
        {
            s.spawn::<_, Result<_, Err>>(move || {
                for step in 0..steps {
                    let step_result: Result<_, Err> = run_step_trailing_group(
                        step,
                        barrier,
                        shared_value,
                        atom_type,
                        group,
                        adder,
                        multiplier,
                        quantum_estimators.as_deref_mut(),
                        debug_estimators.as_deref_mut(),
                        match &mut propagator_and_exchange_potential {
                            Scheme::Regular(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::Regular(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                            Scheme::QuadraticExpansion(SchemeDependent {
                                propagator,
                                exchange_potential,
                            }) => Scheme::QuadraticExpansion(SchemeDependent {
                                propagator: *propagator,
                                exchange_potential: exchange_potential.as_deref_mut(),
                            }),
                        },
                        physical_potential,
                        thermostat,
                        &mut positions,
                        &mut momenta,
                        &mut physical_forces,
                        &mut exchange_forces,
                    );
                    step_result?;

                    barrier.wait();
                }

                Ok(())
            });
        }

        // Main thread.
        for step in 0..steps {
            barrier.wait();
            let physical_potential_energy = main_adder
                .recieve_sum()?
                .expect("All threads should have sent non-empty messages");
            *shared_value.write().map_err(|_| CommError::Main)? = physical_potential_energy;
            barrier.wait();
            let exchange_potential_energy = main_adder
                .recieve_sum()?
                .expect("All threads should have sent non-empty messages");
            *shared_value.write().map_err(|_| CommError::Main)? = exchange_potential_energy;
            barrier.wait();
            let kinetic_energy = main_adder
                .recieve_sum()?
                .expect("All threads should have sent non-empty messages");
            *shared_value.write().map_err(|_| CommError::Main)? = kinetic_energy;
            barrier.wait();

            match main_estimators.as_deref_mut() {
                ObservablesOutputOption::None => {}
                ObservablesOutputOption::Quantum(Estimators { estimators, stream }) => {
                    stream.write_step(step)?;
                    for estimator in estimators {
                        stream.write_value(estimator.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    stream.new_line()?;
                }
                ObservablesOutputOption::Classical(Estimators { estimators, stream }) => {
                    stream.write_step(step)?;
                    for estimator in estimators {
                        stream.write_value(estimator.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    stream.new_line()?;
                }
                ObservablesOutputOption::Shared {
                    quantum_estimators,
                    debug_estimators,
                    stream,
                } => {
                    stream.write_step(step)?;
                    for estimator in quantum_estimators {
                        stream.write_value(estimator.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    for estimator in debug_estimators {
                        stream.write_value(estimator.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    stream.new_line()?;
                }
                ObservablesOutputOption::Separate { quantum, debug } => {
                    quantum.stream.write_step(step)?;
                    for estimator in quantum.estimators {
                        quantum
                            .stream
                            .write_value(estimator.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    quantum.stream.new_line()?;

                    debug.stream.write_step(step)?;
                    for estimator in debug.estimators {
                        debug
                            .stream
                            .write_value(estimator.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    debug.stream.new_line()?;
                }
            }

            barrier.wait();
        }

        Ok(())
    })
}
