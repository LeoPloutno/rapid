#![feature(new_range_api)]
// #![warn(missing_docs)]
use std::{
    fmt::Display,
    iter,
    ops::{Add, Div, Mul},
    sync::{Arc, Barrier, RwLock},
    thread,
};

use arc_rw_lock::{ElementRwLock, UniqueArcElementRwLock, UniqueArcSliceRwLock};

use crate::{
    core::{
        AtomGroupInfo, CommError, DynFactory, DynFullFactory, DynReplicasFactory, Factory, FullFactory,
        HomogeneousFactory, ReplicasFactory,
    },
    observable::{
        debug::{InnerDebugObservable, LeadingDebugObservable, MainDebugObservable, TrailingDebugObservable},
        quantum::{InnerQuantumObservable, LeadingQuantumObservable, MainQuantumObservable, TrailingQuantumObservable},
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

#[allow(dead_code)]
fn simulate_output<const N: usize, T, V, S, A, M, D, B, O, ThermoErr, Err>(
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
        &mut [Box<dyn MainQuantumObservable<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
        &mut [Box<dyn MainDebugObservable<T, V, D, B, A, M, Output = O, Error = impl Into<Err>>>],
        &mut dyn ObservableOutput<O, Error = impl Into<Err>>,
    >,
    propagator: &mut dyn LeadingPropagator<T, V, D, B, ThermoErr, Error = impl Into<Err> + From<ThermoErr>>,
    physical_potential: &mut dyn PhysicalPotential<T, V>,
    exchange_potential: &mut Stat<D, B>,
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
                exchange_potential,
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
        *shared_value.write().map_err(|_| CommError {
            replica: 0,
            group_id: group_idx,
        })? = physical.clone();
        barrier.wait();
        // Other threads read the value.
        barrier.wait();
        let exchange = match adder.recieve_sum().map_err(|err| err.into())? {
            Some(exchange_other_groups) => exchange_other_groups + exchange_group,
            None => exchange_group,
        };
        *shared_value.write().map_err(|_| CommError {
            replica: 0,
            group_id: group_idx,
        })? = exchange.clone();
        barrier.wait();
        // Other threads read the value.
        barrier.wait();
        let kinetic = match adder.recieve_sum().map_err(|err| err.into())? {
            Some(kinetic_other_groups) => kinetic_other_groups + kinetic_group,
            None => kinetic_group,
        };
        *shared_value.write().map_err(|_| CommError {
            replica: 0,
            group_id: group_idx,
        })? = kinetic.clone();
        barrier.wait();

        // Output.
        if let Some(stream) = positions_out.as_deref_mut() {
            let guard = positions.read().as_ref().read_whole().map_err(|_| CommError {
                replica: 0,
                group_id: group_idx,
            })?;
            stream.write(step, groups, &*guard).map_err(|err| err.into())?;
        }
        if let Some(stream) = momenta_out.as_deref_mut() {
            let guard = momenta.read().as_ref().read_whole().map_err(|_| CommError {
                replica: 0,
                group_id: group_idx,
            })?;
            stream.write(step, groups, &*guard).map_err(|err| err.into())?;
        }
        if let Some(stream) = physical_forces_out.as_deref_mut() {
            let guard = physical_forces.read().as_ref().read_whole().map_err(|_| CommError {
                replica: 0,
                group_id: group_idx,
            })?;
            stream.write(step, groups, &*guard).map_err(|err| err.into())?;
        }
        if let Some(stream) = exchange_forces_out.as_deref_mut() {
            let guard = exchange_forces.read().as_ref().read_whole().map_err(|_| CommError {
                replica: 0,
                group_id: group_idx,
            })?;
            stream.write(step, groups, &*guard).map_err(|err| err.into())?;
        }

        macro_rules! write_observables {
            (@quantum $observables:expr, $stream:expr) => {
                for observable in ($observables).iter_mut() {
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
                for observable in ($observables).iter_mut() {
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

#[allow(dead_code)]
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
    exchange_potential: &mut Stat<D, B>,
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
                exchange_potential,
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
            replica: 0,
            group_id: group_idx,
        })?;
        barrier.wait();
        adder.send(exchange_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let exchange = shared_value.read().map_err(|_| CommError {
            replica: 0,
            group_id: group_idx,
        })?;
        barrier.wait();
        adder.send(kinetic_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let kinetic = shared_value.read().map_err(|_| CommError {
            replica: 0,
            group_id: group_idx,
        })?;

        // Output.
        macro_rules! calculate_observables {
            (@quantum $observables:expr) => {
                for observable in ($observables).iter_mut() {
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
                for observable in ($observables).iter_mut() {
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

#[allow(dead_code)]
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
    exchange_potential: &mut Stat<D, B>,
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
                exchange_potential,
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
            replica,
            group_id: group_idx,
        })?;
        barrier.wait();
        adder.send(exchange_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let exchange = shared_value.read().map_err(|_| CommError {
            replica,
            group_id: group_idx,
        })?;
        barrier.wait();
        adder.send(kinetic_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let kinetic = shared_value.read().map_err(|_| CommError {
            replica,
            group_id: group_idx,
        })?;

        // Output.
        macro_rules! calculate_observables {
            (@quantum $observables:expr) => {
                for observable in ($observables).iter_mut() {
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
                for observable in ($observables).iter_mut() {
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

#[allow(dead_code)]
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
    exchange_potential: &mut Stat<D, B>,
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
                exchange_potential,
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
            group_id: group_idx,
        })?;
        barrier.wait();
        adder.send(exchange_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let exchange = shared_value.read().map_err(|_| CommError {
            replica: last_replica,
            group_id: group_idx,
        })?;
        barrier.wait();
        adder.send(kinetic_group).map_err(|err| err.into())?;
        // Output thread writes the value.
        barrier.wait();
        let kinetic = shared_value.read().map_err(|_| CommError {
            replica: last_replica,
            group_id: group_idx,
        })?;

        // Output.
        macro_rules! calculate_observables {
            (@quantum $observables:expr) => {
                for observable in ($observables).iter_mut() {
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
                for observable in ($observables).iter_mut() {
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

fn run<'a, const N: usize, T, V, S, AR, AS, MR, MS, DL, DI, DT, BL, BI, BT, O, ObsErr, ThermoErr, PropErr, Err>(
    steps: usize,
    replicas: usize,
    groups: &'a [AtomGroupInfo<T>],
    adder_senders: &mut [impl Factory<Leading = AS, Inner = impl Iterator<Item = AS>, Trailing = AS>],
    adder_reciever: &mut AR,
    adders: &mut (
             impl FullFactory<
        usize,
        Main = AR,
        Leading: Iterator<Item = AS>,
        Inner: Iterator<Item: Iterator<Item = AS>>,
        Trailing: Iterator<Item = AS>,
    > + ?Sized
         ),
    multiplier_senders: &mut [impl Factory<Leading = MS, Inner = impl Iterator<Item = MS>, Trailing = MS>],
    multiplier_reciever: &mut MR,
    multipliers: &mut (
             impl FullFactory<
        usize,
        Main = MR,
        Leading: Iterator<Item = MS>,
        Inner: Iterator<Item: Iterator<Item = MS>>,
        Trailing: Iterator<Item = MS>,
    > + ?Sized
         ),
    positions_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    momenta_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    physical_forces_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    exchange_forces_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    observables: ObservableOutputOption<
        &mut [Box<
            DynFullFactory<
                &'a [AtomGroupInfo<T>],
                Box<dyn MainQuantumObservable<T, V, AR, MR, Output = O, Error = ObsErr> + Send>,
                Box<dyn LeadingQuantumObservable<T, V, DL, BL, AS, MS, Output = O, Error = ObsErr> + Send>,
                Box<dyn InnerQuantumObservable<T, V, DI, BI, AS, MS, Output = O, Error = ObsErr> + Send>,
                Box<dyn TrailingQuantumObservable<T, V, DT, BT, AS, MS, Output = O, Error = ObsErr> + Send>,
            >,
        >],
        &mut [Box<
            DynFullFactory<
                &'a [AtomGroupInfo<T>],
                Box<dyn MainDebugObservable<T, V, AR, MR, Output = O, Error = ObsErr> + Send>,
                Box<dyn LeadingDebugObservable<T, V, DL, BL, AS, MS, Output = O, Error = ObsErr> + Send>,
                Box<dyn InnerDebugObservable<T, V, DI, BI, AS, MS, Output = O, Error = ObsErr> + Send>,
                Box<dyn TrailingDebugObservable<T, V, DT, BT, AS, MS, Output = O, Error = ObsErr> + Send>,
            >,
        >],
        &mut dyn ObservableOutput<O, Error = impl Into<Err>>,
    >,
    propagators: &mut DynReplicasFactory<
        &'a [AtomGroupInfo<T>],
        Box<dyn LeadingPropagator<T, V, DL, BL, ThermoErr, Error = PropErr> + Send>,
        Box<dyn InnerPropagator<T, V, DI, BI, ThermoErr, Error = PropErr> + Send>,
        Box<dyn TrailingPropagator<T, V, DT, BT, ThermoErr, Error = PropErr> + Send>,
    >,
    physical_potentials: &mut DynReplicasFactory<
        &'a [AtomGroupInfo<T>],
        Box<dyn PhysicalPotential<T, V> + Send>,
        Box<dyn PhysicalPotential<T, V> + Send>,
        Box<dyn PhysicalPotential<T, V> + Send>,
    >,
    exchange_potentials: &mut (
             impl ReplicasFactory<
        &'a [AtomGroupInfo<T>],
        Leading: Iterator<Item = Stat<DL, BL>>,
        Inner: Iterator<Item: Iterator<Item = Stat<DI, BI>>>,
        Trailing: Iterator<Item = Stat<DT, BT>>,
    > + ?Sized
         ),
    thermostats: &mut DynReplicasFactory<
        &'a [AtomGroupInfo<T>],
        Box<dyn Thermostat<T, V, Error = ThermoErr> + Send>,
        Box<dyn Thermostat<T, V, Error = ThermoErr> + Send>,
        Box<dyn Thermostat<T, V, Error = ThermoErr> + Send>,
    >,
    groups_positions: impl ReplicasFactory<
        &'a [AtomGroupInfo<T>],
        Leading: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>,
        Inner: Iterator<Item: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>>,
        Trailing: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>,
    >,
    groups_momenta: impl ReplicasFactory<
        &'a [AtomGroupInfo<T>],
        Leading: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>,
        Inner: Iterator<Item: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>>,
        Trailing: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>,
    >,
    groups_physical_forces: impl ReplicasFactory<
        &'a [AtomGroupInfo<T>],
        Leading: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>,
        Inner: Iterator<Item: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>>,
        Trailing: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>,
    >,
    groups_exchange_forces: impl ReplicasFactory<
        &'a [AtomGroupInfo<T>],
        Leading: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>,
        Inner: Iterator<Item: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>>,
        Trailing: Iterator<Item = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>>,
    >,
) -> Result<(), Err>
where
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Display + Send + Sync,
    V: Vector<N, Element = T> + Display + Send,
    AR: SyncAddRecv<T>,
    AS: SyncAddSend<T> + Send,
    MR: SyncMulRecv<T>,
    MS: SyncMulSend<T> + Send,
    DL: LeadingQuadraticExpansionExchangePotential<T, V> + Distinguishable + Send,
    DI: InnerQuadraticExpansionExchangePotential<T, V> + Distinguishable + Send,
    DT: TrailingQuadraticExpansionExchangePotential<T, V> + Distinguishable + Send,
    BL: LeadingQuadraticExpansionExchangePotential<T, V> + Bosonic + Send,
    BI: InnerQuadraticExpansionExchangePotential<T, V> + Bosonic + Send,
    BT: TrailingQuadraticExpansionExchangePotential<T, V> + Bosonic + Send,
    PropErr: From<ThermoErr>,
    Err: From<CommError<T>> + From<AS::Error> + From<ObsErr> + From<PropErr> + Send,
{
    let (quantum_observables, debug_observables, streams) = observables.split();
    let mut quantum_observables_it = quantum_observables.map(|observables| observables.iter_mut());
    let mut debug_observables_it = debug_observables.map(|observables| observables.iter_mut());

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

    let barrier = Barrier::new(replicas + 1);
    let shared_value = RwLock::new(T::default());

    let barrier = &barrier;
    let shared_value = &shared_value;

    thread::scope(|s| {
        for zip_items!(
            group,
            (mut leading_adder, inner_adders, mut trailing_adder),
            (mut leading_multiplier, inner_multipliers, mut trailing_multiplier),
            replicas_group_quantum_observables,
            replicas_group_debug_observables,
            (mut leading_propagator, inner_propagators, mut trailing_propagator),
            (mut leading_physical_potential, inner_physical_potentials, mut trailing_physical_potential),
            (mut leading_exchange_potential, inner_exchange_potentials, mut trailing_exchange_potential),
            (mut leading_thermostat, inner_thermostats, mut trailing_thermostat),
            (mut first_replica_positions, inner_replicas_positions, mut last_replica_positions),
            (mut first_replica_momenta, inner_replicas_momenta, mut last_replica_momenta),
            (mut first_replica_physical_forces, inner_replicas_physical_forces, mut last_replica_physical_forces),
            (mut first_replica_exchange_forces, inner_replicas_exchange_forces, mut last_replica_exchange_forces)
        ) in zip_iterators!(
            groups.iter(),
            adder_senders.iter_mut().map(|elem| elem.produce()),
            multiplier_senders.iter_mut().map(|elem| elem.produce()),
            iter::from_fn(|| match &mut quantum_observables_it {
                Some(it) => it.next().map(|elem| Some(&mut **elem)),
                None => Some(None),
            }),
            iter::from_fn(|| match &mut debug_observables_it {
                Some(it) => it.next().map(|elem| Some(&mut **elem)),
                None => Some(None),
            }),
            propagators.iter_mut().map(|elem| elem.produce()),
            physical_potentials.iter_mut().map(|elem| elem.produce()),
            exchange_potentials.iter_mut().map(|elem| elem.produce()),
            thermostats.iter_mut().map(|elem| elem.produce()),
            groups_positions.into_iter().map(|mut elem| elem.produce()),
            groups_momenta.into_iter().map(|mut elem| elem.produce()),
            groups_physical_forces.into_iter().map(|mut elem| elem.produce()),
            groups_exchange_forces.into_iter().map(|mut elem| elem.produce())
        ) {
            let (
                main_quantum_observables,
                mut leading_quantum_observables,
                mut inner_quantum_observables_its,
                mut trailing_quantum_observables,
            ) = match replicas_group_quantum_observables {
                Some(observables) => {
                    let mut main_observables = Box::new_uninit_slice(observables.len());
                    let mut leading_observables = Box::new_uninit_slice(observables.len());
                    let mut inner_observables_its = Box::new_uninit_slice(observables.len());
                    let mut trailing_observables = Box::new_uninit_slice(observables.len());

                    for zip_items!(
                        main_observable,
                        leading_observable,
                        inner_observables_it,
                        trailing_observable,
                        observable_factory
                    ) in zip_iterators!(
                        main_observables.iter_mut(),
                        leading_observables.iter_mut(),
                        inner_observables_its.iter_mut(),
                        trailing_observables.iter_mut(),
                        observables
                    ) {
                        let (main, leading, inner, trailing) = observable_factory.produce();
                        main_observable.write(main);
                        leading_observable.write(leading);
                        inner_observables_it.write(inner);
                        trailing_observable.write(trailing);
                    }

                    // SAFETY: Initialized all elements above.
                    unsafe {
                        (
                            Some(main_observables.assume_init()),
                            Some(leading_observables.assume_init()),
                            Some(inner_observables_its.assume_init()),
                            Some(trailing_observables.assume_init()),
                        )
                    }
                }
                None => (None, None, None, None),
            };
            let (
                main_debug_observables,
                mut leading_debug_observables,
                mut inner_debug_observables_its,
                mut trailing_debug_observables,
            ) = match replicas_group_debug_observables {
                Some(observables) => {
                    let mut main_observables = Box::new_uninit_slice(observables.len());
                    let mut leading_observables = Box::new_uninit_slice(observables.len());
                    let mut inner_observables_its = Box::new_uninit_slice(observables.len());
                    let mut trailing_observables = Box::new_uninit_slice(observables.len());

                    for zip_items!(
                        main_observable,
                        leading_observable,
                        inner_observables_it,
                        trailing_observable,
                        observable_factory
                    ) in zip_iterators!(
                        main_observables.iter_mut(),
                        leading_observables.iter_mut(),
                        inner_observables_its.iter_mut(),
                        trailing_observables.iter_mut(),
                        observables
                    ) {
                        let (main, leading, inner, trailing) = observable_factory.produce();
                        main_observable.write(main);
                        leading_observable.write(leading);
                        inner_observables_it.write(inner);
                        trailing_observable.write(trailing);
                    }

                    // SAFETY: Initialized all elements above.
                    unsafe {
                        (
                            Some(main_observables.assume_init()),
                            Some(leading_observables.assume_init()),
                            Some(inner_observables_its.assume_init()),
                            Some(trailing_observables.assume_init()),
                        )
                    }
                }
                None => (None, None, None, None),
            };

            let leading_shared_value_ref = &shared_value;
            s.spawn::<_, Result<_, Err>>(move || {
                let positions = first_replica_positions.as_mut();
                let momenta = first_replica_momenta.as_mut();
                let physical_forces = first_replica_physical_forces.as_mut();
                let exchange_forces = first_replica_exchange_forces.as_mut();

                for step in 0..steps {
                    barrier.wait();

                    let (group_physical_potential_energy, group_exchange_potential_energy) = leading_propagator
                        .propagate(
                            step,
                            &mut *leading_physical_potential,
                            &mut leading_exchange_potential,
                            &mut *leading_thermostat,
                            positions,
                            momenta,
                            physical_forces,
                            exchange_forces,
                        )?;

                    let group_kinetic_energy = momenta
                        .read()
                        .as_ref()
                        .read()
                        .iter()
                        .fold(T::default(), |accum, momentum| {
                            accum + T::from(0.5) * momentum.magnitude_squared() / group.mass.clone()
                        });

                    leading_adder.send(group_physical_potential_energy)?;
                    // Main thread writes the final value.
                    barrier.wait();
                    let physical_potential_energy = leading_shared_value_ref
                        .read()
                        .map_err(|_| CommError {
                            replica: 0,
                            group: group.clone(),
                        })?
                        .clone();
                    barrier.wait();

                    leading_adder.send(group_exchange_potential_energy)?;
                    // Main thread writes the final value.
                    barrier.wait();
                    let exchange_potential_energy = leading_shared_value_ref
                        .read()
                        .map_err(|_| CommError {
                            replica: 0,
                            group: group.clone(),
                        })?
                        .clone();

                    barrier.wait();
                    leading_adder.send(group_kinetic_energy)?;
                    // Main thread writes the final value.
                    barrier.wait();
                    let kinetic_energy = leading_shared_value_ref
                        .read()
                        .map_err(|_| CommError {
                            replica: 0,
                            group: group.clone(),
                        })?
                        .clone();

                    // Output.
                    match &mut leading_quantum_observables {
                        Some(observables) => {
                            for observable in observables.iter_mut() {
                                observable.calculate(
                                    &leading_exchange_potential,
                                    &mut leading_adder,
                                    &mut leading_multiplier,
                                    physical_potential_energy.clone(),
                                    exchange_potential_energy.clone(),
                                    positions,
                                    physical_forces,
                                    exchange_forces,
                                )?;
                            }
                        }
                        None => {}
                    }
                    match &mut leading_debug_observables {
                        Some(observables) => {
                            for observable in observables.iter_mut() {
                                observable.calculate(
                                    &leading_exchange_potential,
                                    &mut leading_adder,
                                    &mut leading_multiplier,
                                    physical_potential_energy.clone(),
                                    exchange_potential_energy.clone(),
                                    kinetic_energy.clone(),
                                    positions,
                                    momenta,
                                    physical_forces,
                                    exchange_forces,
                                )?;
                            }
                        }
                        None => {}
                    }
                }

                Ok(())
            });

            for zip_items!(
                mut adder,
                mut multiplier,
                mut quantum_observables,
                mut debug_observables,
                mut propagator,
                mut physical_potential,
                mut exchange_potential,
                mut thermostat,
                mut positions,
                mut momenta,
                mut physical_forces,
                mut exchange_forces
            ) in zip_iterators!(
                inner_adders,
                inner_multipliers,
                iter::from_fn(|| {
                    match &mut inner_quantum_observables_its {
                        Some(its) => {
                            let mut replica_observables = Vec::with_capacity(its.len());
                            for it in its.iter_mut() {
                                match it.next() {
                                    Some(observable) => replica_observables.push(observable),
                                    None => return None,
                                }
                            }
                            Some(Some(replica_observables.into_boxed_slice()))
                        }
                        None => Some(None),
                    }
                }),
                iter::from_fn(|| {
                    match &mut inner_debug_observables_its {
                        Some(its) => {
                            let mut replica_observables = Vec::with_capacity(its.len());
                            for it in its.iter_mut() {
                                match it.next() {
                                    Some(observable) => replica_observables.push(observable),
                                    None => return None,
                                }
                            }
                            Some(Some(replica_observables.into_boxed_slice()))
                        }
                        None => Some(None),
                    }
                }),
                inner_propagators,
                inner_physical_potentials,
                inner_exchange_potentials,
                inner_thermostats,
                inner_replicas_positions,
                inner_replicas_momenta,
                inner_replicas_physical_forces,
                inner_replicas_exchange_forces
            ) {
                let inner_shared_value_ref = &shared_value;
                s.spawn::<_, Result<_, Err>>(move || {
                    let positions = positions.as_mut();
                    let momenta = momenta.as_mut();
                    let physical_forces = physical_forces.as_mut();
                    let exchange_forces = exchange_forces.as_mut();

                    for step in 0..steps {
                        barrier.wait();

                        let (group_physical_potential_energy, group_exchange_potential_energy) = propagator.propagate(
                            step,
                            &mut *physical_potential,
                            &mut exchange_potential,
                            &mut *thermostat,
                            positions,
                            momenta,
                            physical_forces,
                            exchange_forces,
                        )?;

                        let group_kinetic_energy = momenta
                            .read()
                            .as_ref()
                            .read()
                            .iter()
                            .fold(T::default(), |accum, momentum| {
                                accum + T::from(0.5) * momentum.magnitude_squared() / group.mass.clone()
                            });

                        adder.send(group_physical_potential_energy)?;
                        // Main thread writes the final value.
                        barrier.wait();
                        let physical_potential_energy = inner_shared_value_ref
                            .read()
                            .map_err(|_| CommError {
                                replica: 0,
                                group: group.clone(),
                            })?
                            .clone();
                        barrier.wait();

                        adder.send(group_exchange_potential_energy)?;
                        // Main thread writes the final value.
                        barrier.wait();
                        let exchange_potential_energy = inner_shared_value_ref
                            .read()
                            .map_err(|_| CommError {
                                replica: 0,
                                group: group.clone(),
                            })?
                            .clone();

                        barrier.wait();
                        adder.send(group_kinetic_energy)?;
                        // Main thread writes the final value.
                        barrier.wait();
                        let kinetic_energy = inner_shared_value_ref
                            .read()
                            .map_err(|_| CommError {
                                replica: 0,
                                group: group.clone(),
                            })?
                            .clone();

                        // Output.
                        match &mut quantum_observables {
                            Some(observables) => {
                                for observable in observables.iter_mut() {
                                    observable.calculate(
                                        &exchange_potential,
                                        &mut adder,
                                        &mut multiplier,
                                        physical_potential_energy.clone(),
                                        exchange_potential_energy.clone(),
                                        positions,
                                        physical_forces,
                                        exchange_forces,
                                    )?;
                                }
                            }
                            None => {}
                        }
                        match &mut debug_observables {
                            Some(observables) => {
                                for observable in observables.iter_mut() {
                                    observable.calculate(
                                        &exchange_potential,
                                        &mut adder,
                                        &mut multiplier,
                                        physical_potential_energy.clone(),
                                        exchange_potential_energy.clone(),
                                        kinetic_energy.clone(),
                                        positions,
                                        momenta,
                                        physical_forces,
                                        exchange_forces,
                                    )?;
                                }
                            }
                            None => {}
                        }
                    }

                    Ok(())
                });
            }

            let trailing_shared_value_ref = &shared_value;
            s.spawn::<_, Result<_, Err>>(move || {
                let positions = last_replica_positions.as_mut();
                let momenta = last_replica_momenta.as_mut();
                let physical_forces = last_replica_physical_forces.as_mut();
                let exchange_forces = last_replica_exchange_forces.as_mut();

                for step in 0..steps {
                    barrier.wait();

                    let (group_physical_potential_energy, group_exchange_potential_energy) = trailing_propagator
                        .propagate(
                            step,
                            &mut *trailing_physical_potential,
                            &mut trailing_exchange_potential,
                            &mut *trailing_thermostat,
                            positions,
                            momenta,
                            physical_forces,
                            exchange_forces,
                        )?;

                    let group_kinetic_energy = momenta
                        .read()
                        .as_ref()
                        .read()
                        .iter()
                        .fold(T::default(), |accum, momentum| {
                            accum + T::from(0.5) * momentum.magnitude_squared() / group.mass.clone()
                        });

                    trailing_adder.send(group_physical_potential_energy)?;
                    // Main thread writes the final value.
                    barrier.wait();
                    let physical_potential_energy = trailing_shared_value_ref
                        .read()
                        .map_err(|_| CommError {
                            replica: 0,
                            group: group.clone(),
                        })?
                        .clone();
                    barrier.wait();

                    trailing_adder.send(group_exchange_potential_energy)?;
                    // Main thread writes the final value.
                    barrier.wait();
                    let exchange_potential_energy = trailing_shared_value_ref
                        .read()
                        .map_err(|_| CommError {
                            replica: 0,
                            group: group.clone(),
                        })?
                        .clone();

                    barrier.wait();
                    trailing_adder.send(group_kinetic_energy)?;
                    // Main thread writes the final value.
                    barrier.wait();
                    let kinetic_energy = trailing_shared_value_ref
                        .read()
                        .map_err(|_| CommError {
                            replica: 0,
                            group: group.clone(),
                        })?
                        .clone();

                    // Output.
                    match &mut trailing_quantum_observables {
                        Some(observables) => {
                            for observable in observables.iter_mut() {
                                observable.calculate(
                                    &trailing_exchange_potential,
                                    &mut trailing_adder,
                                    &mut trailing_multiplier,
                                    physical_potential_energy.clone(),
                                    exchange_potential_energy.clone(),
                                    positions,
                                    physical_forces,
                                    exchange_forces,
                                )?;
                            }
                        }
                        None => {}
                    }
                    match &mut trailing_debug_observables {
                        Some(observables) => {
                            for observable in observables.iter_mut() {
                                observable.calculate(
                                    &trailing_exchange_potential,
                                    &mut trailing_adder,
                                    &mut trailing_multiplier,
                                    physical_potential_energy.clone(),
                                    exchange_potential_energy.clone(),
                                    kinetic_energy.clone(),
                                    positions,
                                    momenta,
                                    physical_forces,
                                    exchange_forces,
                                )?;
                            }
                        }
                        None => {}
                    }
                }

                Ok(())
            });
        }

        todo!()
    })
}
