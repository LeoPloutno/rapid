#![allow(dead_code)]
#![feature(new_range_api)]
// #![warn(missing_docs)]

use std::{
    error::Error,
    fmt::Display,
    io::Write,
    ops::{Add, AddAssign, Div, Mul},
    sync::{Barrier, LockResult, PoisonError},
};

use arc_rw_lock::{ElementRwLock, MappedRwLockReadWholeGuard, UniqueArcSliceRwLock};

use crate::{
    core::AtomGroupInfo,
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
pub mod potential;
pub mod propagator;
pub mod stat;
pub mod sync_ops;
pub mod thermostat;
pub mod vector;

fn simulate_leading<T, const N: usize, V, O, A, M, D, B, E>(
    steps: usize,
    step_size: T,
    replicas: usize,
    replica: usize,
    groups: &[AtomGroupInfo<T>],
    group_idx: usize,
    mut positions_out: Option<&mut O>,
    mut momenta_out: Option<&mut O>,
    mut forces_out: Option<&mut O>,
    slice_barrier: &Barrier,
    group_barrier: &Barrier,
    global_barrier: &Barrier,
    adder: &mut A,
    multiplier: &mut M,
    propagator: &mut dyn LeadingPropagator<T, V, D, B>,
    phys_potential: &mut dyn PhysicalPotential<T, V>,
    groups_exch_potentials: &mut [Stat<D, B>],
    thermostat: &mut dyn Thermostat<T, V>,
    positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
) -> Result<(), E>
where
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T>,
    V: Vector<N, Element = T> + Display,
    O: Write,
    A: SyncAddRecv<T, Error: Error + 'static> + ?Sized,
    M: SyncMulRecv<T, Error: Error + 'static> + ?Sized,
    D: LeadingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: LeadingQuadraticExpansionExchangePotential<T, V> + Bosonic,
    E: From<A::Error> + From<M::Error>,
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
        );

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
        if let Some(stream) = positions_out.take() {
            let guard = positions.read().as_ref().read_whole().unwrap();
            for AtomGroupInfo { span, label, .. } in groups {
                for position in guard
                    .get(*span)
                    .expect("`span` should be a valid span in `positions`")
                {
                    stream.write(format!("{}\t{}\n", label.as_str(), position).as_bytes());
                }
            }
        }
    }
    Ok(())
}
