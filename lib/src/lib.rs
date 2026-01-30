#![allow(dead_code)]
#![feature(new_range_api)]
// #![warn(missing_docs)]

use std::{
    fmt::Display,
    ops::{Add, Div, Mul},
    sync::{Barrier, RwLock},
};

use arc_rw_lock::{ElementRwLock, UniqueArcSliceRwLock};

use crate::{
    core::{AtomGroupInfo, CommError},
    observable::{
        debug::{InnerDebugObservable, LeadingDebugObservable, LeadingDebugObservableOutput, TrailingDebugObservable},
        quantum::{
            InnerQuantumObservable, LeadingQuantumObservable, LeadingQuantumObservableOutput, TrailingQuantumObservable,
        },
    },
    output::{ObservableOption, ObservableOutput, ObservableOutputOption, Observables, VectorsOutput},
    potential::{
        exchange::quadratic::{
            InnerQuadraticExpansionExchangePotential, LeadingQuadraticExpansionExchangePotential,
            TrailingQuadraticExpansionExchangePotential,
        },
        physical::PhysicalPotential,
    },
    propagator::{InnerPropagator, LeadingPropagator, TrailingPropagator},
    stat::{Bosonic, Distinguishable, Stat},
    sync_ops::{SyncAddRecv, SyncAddSend, SyncMulRecv, SyncMulSend},
    thermostat::Thermostat,
    vector::Vector,
};

pub mod core;
pub mod marker;
pub mod observable;
pub mod output;
pub mod potential;
pub mod propagator;
pub mod stat;
pub mod sync_ops;
pub mod thermostat;
pub mod vector;

fn simulate_leading_output<const N: usize, T, V, S, A, M, D, B, O, ThermoErr, Err>(
    steps: usize,
    step_size: T,
    groups: &[AtomGroupInfo<T>],
    group_idx: usize,
    barrier: &Barrier,
    shared_value: &RwLock<T>,
    adder: &mut A,
    multiplier: &mut M,
    mut positions_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    mut momenta_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    mut physical_forces_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    mut exchange_forces_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    mut observables: ObservableOutputOption<
        &mut [Box<dyn LeadingQuantumObservableOutput<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
        &mut [Box<dyn LeadingDebugObservableOutput<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
        &mut dyn ObservableOutput<O, Error = impl Into<Err>>,
    >,
    propagator: &mut dyn LeadingPropagator<T, V, D, B, ThermoErr, Error = impl Into<Err> + From<ThermoErr>>,
    physical_potential: &mut dyn PhysicalPotential<T, V>,
    groups_exchange_potentials: &mut [Stat<D, B>],
    thermostat: &mut dyn Thermostat<T, V, Error = ThermoErr>,
    positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
) -> Result<(), Err>
where
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Display,
    V: Vector<N, Element = T> + Display,
    A: SyncAddRecv<T, Error: Into<Err>> + ?Sized,
    M: SyncMulRecv<T, Error: Into<Err>> + ?Sized,
    D: LeadingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: LeadingQuadraticExpansionExchangePotential<T, V> + Bosonic,
    Err: From<CommError>,
{
    for step in 0..steps {
        barrier.wait();

        let (physical_group, exchange_group) = propagator
            .propagate(
                step,
                step_size.clone(),
                group_idx,
                groups,
                physical_potential,
                groups_exchange_potentials,
                thermostat,
                positions,
                momenta,
                physical_forces,
                exchange_forces,
            )
            .map_err(|err| err.into())?;

        let mass = groups
            .get(group_idx)
            .expect("`group_idx` should be a valid index in `groups`")
            .mass
            .clone();
        let kinetic_group = momenta
            .read()
            .as_ref()
            .read()
            .iter()
            .fold(T::default(), |accum, momentum| {
                accum + T::from(0.5) * momentum.magnitude_squared() / mass.clone()
            });

        let physical = match adder.recieve_sum().map_err(|err| err.into())? {
            Some(physical_other_groups) => physical_other_groups + physical_group,
            None => physical_group,
        };
        *shared_value.write().map_err(|_| CommError { replica: 0, group_idx })? = physical.clone();
        barrier.wait();
        // Other threads read the value.
        barrier.wait();
        let exchange = match adder.recieve_sum().map_err(|err| err.into())? {
            Some(exchange_other_groups) => exchange_other_groups + exchange_group,
            None => exchange_group,
        };
        *shared_value.write().map_err(|_| CommError { replica: 0, group_idx })? = exchange.clone();
        barrier.wait();
        // Other threads read the value.
        barrier.wait();
        let kinetic = match adder.recieve_sum().map_err(|err| err.into())? {
            Some(kinetic_other_groups) => kinetic_other_groups + kinetic_group,
            None => kinetic_group,
        };
        *shared_value.write().map_err(|_| CommError { replica: 0, group_idx })? = kinetic.clone();
        barrier.wait();

        // Output.
        if let Some(stream) = positions_out.as_deref_mut() {
            let guard = positions
                .read()
                .as_ref()
                .read_whole()
                .map_err(|_| CommError { replica: 0, group_idx })?;
            stream.write(step, groups, &*guard).map_err(|err| err.into())?;
        }
        if let Some(stream) = momenta_out.as_deref_mut() {
            let guard = momenta
                .read()
                .as_ref()
                .read_whole()
                .map_err(|_| CommError { replica: 0, group_idx })?;
            stream.write(step, groups, &*guard).map_err(|err| err.into())?;
        }
        if let Some(stream) = physical_forces_out.as_deref_mut() {
            let guard = physical_forces
                .read()
                .as_ref()
                .read_whole()
                .map_err(|_| CommError { replica: 0, group_idx })?;
            stream.write(step, groups, &*guard).map_err(|err| err.into())?;
        }
        if let Some(stream) = exchange_forces_out.as_deref_mut() {
            let guard = exchange_forces
                .read()
                .as_ref()
                .read_whole()
                .map_err(|_| CommError { replica: 0, group_idx })?;
            stream.write(step, groups, &*guard).map_err(|err| err.into())?;
        }

        macro_rules! write_observables {
            (@quantum $observables:expr, $stream:expr) => {
                for (observable, exchange_potential) in ($observables).iter_mut().zip(groups_exchange_potentials.iter())
                {
                    ($stream)
                        .write_observable(
                            observable
                                .calculate(
                                    group_idx,
                                    groups,
                                    exchange_potential,
                                    adder,
                                    multiplier,
                                    positions,
                                    physical_forces,
                                    exchange_forces,
                                )
                                .map_err(|err| err.into())?,
                        )
                        .map_err(|err| err.into())?;
                }
            };
            (@debug $observables:expr, $stream:expr) => {
                for (observable, exchange_potential) in ($observables).iter_mut().zip(groups_exchange_potentials.iter())
                {
                    ($stream)
                        .write_observable(
                            observable
                                .calculate(
                                    group_idx,
                                    groups,
                                    exchange_potential,
                                    adder,
                                    multiplier,
                                    physical.clone(),
                                    exchange.clone(),
                                    kinetic.clone(),
                                    positions,
                                    momenta,
                                    physical_forces,
                                    exchange_forces,
                                )
                                .map_err(|err| err.into())?,
                        )
                        .map_err(|err| err.into())?;
                }
            };
        }

        match &mut observables {
            ObservableOutputOption::None => {}
            ObservableOutputOption::Quantum(Observables { observables, stream }) => {
                stream.write_step(step).map_err(|err| err.into())?;
                write_observables!(@quantum *observables, *stream);
                stream.new_line().map_err(|err| err.into())?;
            }
            ObservableOutputOption::Debug(Observables { observables, stream }) => {
                stream.write_step(step).map_err(|err| err.into())?;
                write_observables!(@debug *observables, *stream);
                stream.new_line().map_err(|err| err.into())?;
            }
            ObservableOutputOption::Separate { quantum, debug } => {
                quantum.stream.write_step(step).map_err(|err| err.into())?;
                write_observables!(@quantum quantum.observables, quantum.stream);
                quantum.stream.new_line().map_err(|err| err.into())?;

                debug.stream.write_step(step).map_err(|err| err.into())?;
                write_observables!(@debug debug.observables, debug.stream);
                debug.stream.new_line().map_err(|err| err.into())?;
            }
            ObservableOutputOption::Shared {
                quantum: quantum_observables,
                debug: debug_observables,
                stream,
            } => {
                stream.write_step(step).map_err(|err| err.into())?;
                write_observables!(@quantum *quantum_observables, *stream);
                write_observables!(@debug *debug_observables, *stream);
                stream.new_line().map_err(|err| err.into())?;
            }
        }
    }
    Ok(())
}

fn simulate_leading<const N: usize, T, V, S, A, M, D, B, O, ThermoErr, Err>(
    steps: usize,
    step_size: T,
    groups: &[AtomGroupInfo<T>],
    group_idx: usize,
    barrier: &Barrier,
    shared_value: &RwLock<T>,
    adder: &mut A,
    multiplier: &mut M,
    mut observables: ObservableOption<
        &mut [Box<dyn LeadingQuantumObservable<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
        &mut [Box<dyn LeadingDebugObservable<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
    >,
    propagator: &mut dyn LeadingPropagator<T, V, D, B, ThermoErr, Error = impl Into<Err> + From<ThermoErr>>,
    physical_potential: &mut dyn PhysicalPotential<T, V>,
    groups_exchange_potentials: &mut [Stat<D, B>],
    thermostat: &mut dyn Thermostat<T, V, Error = ThermoErr>,
    positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
) -> Result<(), Err>
where
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Display,
    V: Vector<N, Element = T> + Display,
    A: SyncAddSend<T, Error: Into<Err>> + ?Sized,
    M: SyncMulSend<T, Error: Into<Err>> + ?Sized,
    D: LeadingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: LeadingQuadraticExpansionExchangePotential<T, V> + Bosonic,
    Err: From<CommError>,
{
    for step in 0..steps {
        barrier.wait();

        let (physical_group, exchange_group) = propagator
            .propagate(
                step,
                step_size.clone(),
                group_idx,
                groups,
                physical_potential,
                groups_exchange_potentials,
                thermostat,
                positions,
                momenta,
                physical_forces,
                exchange_forces,
            )
            .map_err(|err| err.into())?;

        let mass = groups
            .get(group_idx)
            .expect("`group_idx` should be a valid index in `groups`")
            .mass
            .clone();
        let kinetic_group = momenta
            .read()
            .as_ref()
            .read()
            .iter()
            .fold(T::default(), |accum, momentum| {
                accum + T::from(0.5) * momentum.magnitude_squared() / mass.clone()
            });

        adder.send(physical_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let physical = shared_value.read().map_err(|_| CommError { replica: 0, group_idx })?;
        barrier.wait();
        adder.send(exchange_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let exchange = shared_value.read().map_err(|_| CommError { replica: 0, group_idx })?;
        barrier.wait();
        adder.send(kinetic_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let kinetic = shared_value.read().map_err(|_| CommError { replica: 0, group_idx })?;

        // Output.
        macro_rules! calculate_observables {
            (@quantum $observables:expr) => {
                for (observable, exchange_potential) in ($observables).iter_mut().zip(groups_exchange_potentials.iter())
                {
                    observable
                        .calculate(
                            group_idx,
                            groups,
                            exchange_potential,
                            adder,
                            multiplier,
                            positions,
                            physical_forces,
                            exchange_forces,
                        )
                        .map_err(|err| err.into())?
                }
            };
            (@debug $observables:expr) => {
                for (observable, exchange_potential) in ($observables).iter_mut().zip(groups_exchange_potentials.iter())
                {
                    observable
                        .calculate(
                            group_idx,
                            groups,
                            exchange_potential,
                            adder,
                            multiplier,
                            physical.clone(),
                            exchange.clone(),
                            kinetic.clone(),
                            positions,
                            momenta,
                            physical_forces,
                            exchange_forces,
                        )
                        .map_err(|err| err.into())?
                }
            };
        }

        match &mut observables {
            ObservableOption::None => {}
            ObservableOption::Quantum(observables) => calculate_observables!(@quantum *observables),
            ObservableOption::Debug(observables) => calculate_observables!(@debug *observables),
            ObservableOption::All { quantum, debug } => {
                calculate_observables!(@quantum *quantum);
                calculate_observables!(@debug *debug);
            }
        }
    }
    Ok(())
}

fn simulate_inner<const N: usize, T, V, S, A, M, D, B, O, ThermoErr, Err>(
    steps: usize,
    step_size: T,
    replica: usize,
    groups: &[AtomGroupInfo<T>],
    group_idx: usize,
    barrier: &Barrier,
    shared_value: &RwLock<T>,
    adder: &mut A,
    multiplier: &mut M,
    mut observables: ObservableOption<
        &mut [Box<dyn InnerQuantumObservable<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
        &mut [Box<dyn InnerDebugObservable<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
    >,
    propagator: &mut dyn InnerPropagator<T, V, D, B, ThermoErr, Error = impl Into<Err> + From<ThermoErr>>,
    physical_potential: &mut dyn PhysicalPotential<T, V>,
    groups_exchange_potentials: &mut [Stat<D, B>],
    thermostat: &mut dyn Thermostat<T, V, Error = ThermoErr>,
    positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
) -> Result<(), Err>
where
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Display,
    V: Vector<N, Element = T> + Display,
    A: SyncAddSend<T, Error: Into<Err>> + ?Sized,
    M: SyncMulSend<T, Error: Into<Err>> + ?Sized,
    D: InnerQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: InnerQuadraticExpansionExchangePotential<T, V> + Bosonic,
    Err: From<CommError>,
{
    for step in 0..steps {
        barrier.wait();

        let (physical_group, exchange_group) = propagator
            .propagate(
                step,
                step_size.clone(),
                replica,
                group_idx,
                groups,
                physical_potential,
                groups_exchange_potentials,
                thermostat,
                positions,
                momenta,
                physical_forces,
                exchange_forces,
            )
            .map_err(|err| err.into())?;

        let mass = groups
            .get(group_idx)
            .expect("`group_idx` should be a valid index in `groups`")
            .mass
            .clone();
        let kinetic_group = momenta
            .read()
            .as_ref()
            .read()
            .iter()
            .fold(T::default(), |accum, momentum| {
                accum + T::from(0.5) * momentum.magnitude_squared() / mass.clone()
            });

        adder.send(physical_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let physical = shared_value.read().map_err(|_| CommError { replica, group_idx })?;
        barrier.wait();
        adder.send(exchange_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let exchange = shared_value.read().map_err(|_| CommError { replica, group_idx })?;
        barrier.wait();
        adder.send(kinetic_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let kinetic = shared_value.read().map_err(|_| CommError { replica, group_idx })?;

        // Output.
        macro_rules! calculate_observables {
            (@quantum $observables:expr) => {
                for (observable, exchange_potential) in ($observables).iter_mut().zip(groups_exchange_potentials.iter())
                {
                    observable
                        .calculate(
                            replica,
                            group_idx,
                            groups,
                            exchange_potential,
                            adder,
                            multiplier,
                            positions,
                            physical_forces,
                            exchange_forces,
                        )
                        .map_err(|err| err.into())?
                }
            };
            (@debug $observables:expr) => {
                for (observable, exchange_potential) in ($observables).iter_mut().zip(groups_exchange_potentials.iter())
                {
                    observable
                        .calculate(
                            replica,
                            group_idx,
                            groups,
                            exchange_potential,
                            adder,
                            multiplier,
                            physical.clone(),
                            exchange.clone(),
                            kinetic.clone(),
                            positions,
                            momenta,
                            physical_forces,
                            exchange_forces,
                        )
                        .map_err(|err| err.into())?
                }
            };
        }

        match &mut observables {
            ObservableOption::None => {}
            ObservableOption::Quantum(observables) => calculate_observables!(@quantum *observables),
            ObservableOption::Debug(observables) => calculate_observables!(@debug *observables),
            ObservableOption::All { quantum, debug } => {
                calculate_observables!(@quantum *quantum);
                calculate_observables!(@debug *debug);
            }
        }
    }
    Ok(())
}

fn simulate_trailing<const N: usize, T, V, S, A, M, D, B, O, ThermoErr, Err>(
    steps: usize,
    step_size: T,
    last_replica: usize,
    groups: &[AtomGroupInfo<T>],
    group_idx: usize,
    barrier: &Barrier,
    shared_value: &RwLock<T>,
    adder: &mut A,
    multiplier: &mut M,
    mut observables: ObservableOption<
        &mut [Box<dyn TrailingQuantumObservable<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
        &mut [Box<dyn TrailingDebugObservable<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
    >,
    propagator: &mut dyn TrailingPropagator<T, V, D, B, ThermoErr, Error = impl Into<Err> + From<ThermoErr>>,
    physical_potential: &mut dyn PhysicalPotential<T, V>,
    groups_exchange_potentials: &mut [Stat<D, B>],
    thermostat: &mut dyn Thermostat<T, V, Error = ThermoErr>,
    positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
) -> Result<(), Err>
where
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Display,
    V: Vector<N, Element = T> + Display,
    A: SyncAddSend<T, Error: Into<Err>> + ?Sized,
    M: SyncMulSend<T, Error: Into<Err>> + ?Sized,
    D: TrailingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: TrailingQuadraticExpansionExchangePotential<T, V> + Bosonic,
    Err: From<CommError>,
{
    for step in 0..steps {
        barrier.wait();

        let (physical_group, exchange_group) = propagator
            .propagate(
                step,
                step_size.clone(),
                last_replica,
                group_idx,
                groups,
                physical_potential,
                groups_exchange_potentials,
                thermostat,
                positions,
                momenta,
                physical_forces,
                exchange_forces,
            )
            .map_err(|err| err.into())?;

        let mass = groups
            .get(group_idx)
            .expect("`group_idx` should be a valid index in `groups`")
            .mass
            .clone();
        let kinetic_group = momenta
            .read()
            .as_ref()
            .read()
            .iter()
            .fold(T::default(), |accum, momentum| {
                accum + T::from(0.5) * momentum.magnitude_squared() / mass.clone()
            });

        adder.send(physical_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let physical = shared_value.read().map_err(|_| CommError {
            replica: last_replica,
            group_idx,
        })?;
        barrier.wait();
        adder.send(exchange_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let exchange = shared_value.read().map_err(|_| CommError {
            replica: last_replica,
            group_idx,
        })?;
        barrier.wait();
        adder.send(kinetic_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let kinetic = shared_value.read().map_err(|_| CommError {
            replica: last_replica,
            group_idx,
        })?;

        // Output.
        macro_rules! calculate_observables {
            (@quantum $observables:expr) => {
                for (observable, exchange_potential) in ($observables).iter_mut().zip(groups_exchange_potentials.iter())
                {
                    observable
                        .calculate(
                            last_replica,
                            group_idx,
                            groups,
                            exchange_potential,
                            adder,
                            multiplier,
                            positions,
                            physical_forces,
                            exchange_forces,
                        )
                        .map_err(|err| err.into())?
                }
            };
            (@debug $observables:expr) => {
                for (observable, exchange_potential) in ($observables).iter_mut().zip(groups_exchange_potentials.iter())
                {
                    observable
                        .calculate(
                            last_replica,
                            group_idx,
                            groups,
                            exchange_potential,
                            adder,
                            multiplier,
                            physical.clone(),
                            exchange.clone(),
                            kinetic.clone(),
                            positions,
                            momenta,
                            physical_forces,
                            exchange_forces,
                        )
                        .map_err(|err| err.into())?
                }
            };
        }

        match &mut observables {
            ObservableOption::None => {}
            ObservableOption::Quantum(observables) => calculate_observables!(@quantum *observables),
            ObservableOption::Debug(observables) => calculate_observables!(@debug *observables),
            ObservableOption::All { quantum, debug } => {
                calculate_observables!(@quantum *quantum);
                calculate_observables!(@debug *debug);
            }
        }
    }
    Ok(())
}
