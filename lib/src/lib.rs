#![allow(dead_code)]
#![feature(new_range_api)]
// #![warn(missing_docs)]

use std::{
    error::Error,
    fmt::Display,
    io::{Error as IoError, Write},
    ops::{Add, AddAssign, Div, Mul},
    sync::{Barrier, LockResult, PoisonError},
};

use arc_rw_lock::{ElementRwLock, MappedRwLockReadWholeGuard, UniqueArcSliceRwLock};

use crate::{
    core::{AtomGroupInfo, CommError},
    observable::{
        ScalarOrVector, debug::LeadingDebugObservable, quantum::LeadingQuantumObservable,
    },
    output::{Observables, ObservablesOption, ObservablesOutput, VectorsOutput},
    potential::{
        exchange::quadratic::LeadingQuadraticExpansionExchangePotential,
        physical::PhysicalPotential,
    },
    propagator::LeadingPropagator,
    stat::{Bosonic, Distinguishable, Stat},
    sync_ops::{SyncAddRecv, SyncMulRecv},
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

fn simulate_leading<
    const N: usize,
    T,
    V,
    S,
    A,
    M,
    D,
    B,
    VecErr,
    ObsErr,
    ObsOutErr,
    ThermoErr,
    PropErr,
    E,
>(
    steps: usize,
    step_size: T,
    replicas: usize,
    replica: usize,
    groups: &[AtomGroupInfo<T>],
    group_idx: usize,
    mut positions_out: Option<&mut dyn VectorsOutput<N, T, V, Error = VecErr>>,
    mut momenta_out: Option<&mut dyn VectorsOutput<N, T, V, Error = VecErr>>,
    mut forces_out: Option<&mut dyn VectorsOutput<N, T, V, Error = VecErr>>,
    mut observables: ObservablesOption<
        &mut [Box<
            dyn LeadingQuantumObservable<
                    T,
                    V,
                    D,
                    B,
                    A,
                    M,
                    Output = ScalarOrVector<N, T, V>,
                    Error = ObsErr,
                >,
        >],
        &mut [Box<
            dyn LeadingDebugObservable<
                    T,
                    V,
                    D,
                    B,
                    A,
                    M,
                    Output = ScalarOrVector<N, T, V>,
                    Error = ObsErr,
                >,
        >],
        &mut dyn ObservablesOutput<
            N,
            T,
            V,
            ObsErr,
            Input = ScalarOrVector<N, T, V>,
            Error = ObsOutErr,
        >,
    >,
    slice_barrier: &Barrier,
    group_barrier: &Barrier,
    global_barrier: &Barrier,
    adder: &mut A,
    multiplier: &mut M,
    propagator: &mut dyn LeadingPropagator<T, V, D, B, ThermoErr, Error = PropErr>,
    phys_potential: &mut dyn PhysicalPotential<T, V>,
    groups_exch_potentials: &mut [Stat<D, B>],
    thermostat: &mut dyn Thermostat<T, V, Error = ThermoErr>,
    positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
) -> Result<(), E>
where
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Display,
    V: Vector<N, Element = T> + Display,
    A: SyncAddRecv<T, Error: Error + 'static> + ?Sized,
    M: SyncMulRecv<T, Error: Error + 'static> + ?Sized,
    D: LeadingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: LeadingQuadraticExpansionExchangePotential<T, V> + Bosonic,
    ObsOutErr: From<ObsErr>,
    PropErr: From<ThermoErr>,
    E: From<VecErr>
        + From<ObsOutErr>
        + From<ThermoErr>
        + From<PropErr>
        + From<CommError>
        + From<A::Error>
        + From<M::Error>
        + From<IoError>,
{
    for step in 0..steps {
        let (physical, exchange) = propagator.propagate(
            step,
            step_size.clone(),
            groups,
            group_idx,
            phys_potential,
            groups_exch_potentials,
            thermostat,
            positions,
            momenta,
            forces,
        )?;

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
        let kinetic = match adder.recieve_sum()? {
            Some(kinetic_other_groups) => kinetic_other_groups + kinetic_group,
            None => kinetic_group,
        };

        global_barrier.wait();

        // Output.
        if let Some(stream) = positions_out.as_deref_mut() {
            let guard = positions
                .read()
                .as_ref()
                .read_whole()
                .map_err(|_| CommError { replica, group_idx })?;
            stream.write(step, groups, &*guard)?;
        }
        if let Some(stream) = momenta_out.as_deref_mut() {
            let guard = momenta
                .read()
                .as_ref()
                .read_whole()
                .map_err(|_| CommError { replica, group_idx })?;
            stream.write(step, groups, &*guard)?;
        }
        if let Some(stream) = forces_out.as_deref_mut() {
            let guard = forces
                .read()
                .as_ref()
                .read_whole()
                .map_err(|_| CommError { replica, group_idx })?;
            stream.write(step, groups, &*guard)?;
        }

        match &mut observables {
            ObservablesOption::None => {}
            ObservablesOption::Quantum(Observables {
                observables,
                stream,
            }) => stream.write(step, groups, observables.iter_mut().map(|observable| observable.calculate(adder, multiplier, exchange_potential, groups, group_idx, positions, forces)),
            _ => todo!(),
        }
    }

    Ok(())
}
