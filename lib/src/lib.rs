#![feature(new_range_api)]
// #![warn(missing_docs)]
use std::{
    fmt::Display,
    iter,
    ops::{Add, Div, Mul},
    sync::{Arc, Barrier, RwLock},
    thread,
};

use arc_rw_lock::{UniqueArcElementRwLock, UniqueArcSliceRwLock};

use crate::{
    core::{AtomType, CommError, Factory, FullFactory},
    observable::{
        debug::{InnerDebugObservable, LeadingDebugObservable, MainDebugObservable, TrailingDebugObservable},
        quantum::{InnerQuantumObservable, LeadingQuantumObservable, MainQuantumObservable, TrailingQuantumObservable},
    },
    output::{ObservableOutput, ObservableOutputOption, VectorsOutput},
    potential::{
        exchange::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential},
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

fn run<
    const N: usize,
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Display + Send + Sync,
    V: Vector<N, Element = T> + Display + Send,
    AdderReciever: SyncAddRecv<T> + ?Sized,
    AdderSender: SyncAddSend<T> + ?Sized,
    MultiplierReciever: SyncMulRecv<T> + ?Sized,
    MultiplierSender: SyncMulSend<T> + ?Sized,
    QObsMain: MainQuantumObservable<T, V, AdderReciever, MultiplierReciever, Output = Output> + ?Sized,
    QObsLeading: LeadingQuantumObservable<T, V, AdderSender, MultiplierSender, DistLeading, BosonLeading, Output = Output>,
    QObsInner: InnerQuantumObservable<T, V, AdderSender, MultiplierSender, DistInner, BosonInner, Output = Output>,
    QObsTrailing: TrailingQuantumObservable<T, V, AdderSender, MultiplierSender, DistTrailing, BosonTrailing, Output = Output>,
    DObsMain: MainDebugObservable<T, V, AdderReciever, MultiplierReciever, Output = Output> + ?Sized,
    DObsLeading: LeadingDebugObservable<T, V, AdderSender, MultiplierSender, DistLeading, BosonLeading, Output = Output> + ?Sized,
    DObsInner: InnerDebugObservable<T, V, AdderSender, MultiplierSender, DistInner, BosonInner, Output = Output> + ?Sized,
    DObsTrailing: TrailingDebugObservable<T, V, AdderSender, MultiplierSender, DistTrailing, BosonTrailing, Output = Output> + ?Sized,
    ObsOut: ObservableOutput<Output> + ?Sized,
    PropLeading: LeadingPropagator<T, V, Phys, DistLeading, BosonLeading, Therm> + Send + ?Sized,
    PropInner: InnerPropagator<T, V, Phys, DistInner, BosonInner, Therm> + Send + ?Sized,
    PropTrailing: TrailingPropagator<T, V, Phys, DistTrailing, BosonTrailing, Therm> + Send + ?Sized,
    Phys: PhysicalPotential<T, V> + Send + ?Sized,
    DistLeading: LeadingExchangePotential<T, V> + Distinguishable + Send,
    DistInner: InnerExchangePotential<T, V> + Distinguishable + Send,
    DistTrailing: TrailingExchangePotential<T, V> + Distinguishable + Send,
    BosonLeading: LeadingExchangePotential<T, V> + Bosonic + Send,
    BosonInner: InnerExchangePotential<T, V> + Bosonic + Send,
    BosonTrailing: TrailingExchangePotential<T, V> + Bosonic + Send,
    Therm: Thermostat<T, V> + Send + ?Sized,
    Output,
    Err: From<CommError<T>>
        + From<AdderSender::Error>
        + From<PropLeading::Error>
        + From<QObsMain::Error>
        + From<QObsLeading::Error>
        + From<QObsInner::Error>
        + From<QObsTrailing::Error>
        + From<DObsMain::Error>
        + From<DObsLeading::Error>
        + From<DObsInner::Error>
        + From<DObsTrailing::Error>
        + Send,
>(
    steps: usize,
    replicas: usize,
    groups: &[AtomType<T>],
    adders: &mut (
             impl for<'a> FullFactory<
        'a,
        T,
        Main = AdderReciever,
        Leading = AdderSender,
        Inner = AdderSender,
        Trailing = AdderSender,
    > + ?Sized
         ),
    multipliers: &mut (
             impl for<'a> FullFactory<
        'a,
        T,
        Main = MultiplierReciever,
        Leading = MultiplierSender,
        Inner = MultiplierSender,
        Trailing = MultiplierSender,
    > + ?Sized
         ),
    positions_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    momenta_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    physical_forces_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    exchange_forces_out: Option<&mut dyn VectorsOutput<N, T, V, Error = impl Into<Err>>>,
    observables: ObservableOutputOption<
        &mut [impl for<'a> FullFactory<
            'a,
            T,
            Main = QObsMain,
            Leading = QObsLeading,
            Inner = QObsInner,
            Trailing = QObsTrailing,
        >],
        &mut [impl for<'a> FullFactory<
            'a,
            T,
            Main = DObsMain,
            Leading = DObsLeading,
            Inner = DObsInner,
            Trailing = DObsTrailing,
        >],
        &mut ObsOut,
    >,
    propagators: &mut (
             impl for<'a> Factory<'a, T, Leading = PropLeading, Inner = PropInner, Trailing = PropTrailing> + ?Sized
         ),
    physical_potentials: &mut (impl for<'a> Factory<'a, T, Leading = Phys, Inner = Phys, Trailing = Phys> + ?Sized),
    exchange_potentials: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Leading = Stat<DistLeading, BosonLeading>,
        Inner = Stat<DistInner, BosonInner>,
        Trailing = Stat<DistTrailing, BosonTrailing>,
    > + ?Sized
         ),
    thermostats: &mut (impl for<'a> Factory<'a, T, Leading = Therm, Inner = Therm, Trailing = Therm> + ?Sized),
    positions: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Inner = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
        Inner = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
        Trailing = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
    > + ?Sized
         ),
    momenta: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Inner = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
        Inner = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
        Trailing = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
    > + ?Sized
         ),
    physical_forces: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Inner = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
        Inner = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
        Trailing = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
    > + ?Sized
         ),
    exchange_forces: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Inner = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
        Inner = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
        Trailing = UniqueArcElementRwLock<UniqueArcSliceRwLock<V>>,
    > + ?Sized
         ),
) -> Result<(), Err> {
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
            positions.into_iter().map(|mut elem| elem.produce()),
            momenta.into_iter().map(|mut elem| elem.produce()),
            physical_forces.into_iter().map(|mut elem| elem.produce()),
            exchange_forces.into_iter().map(|mut elem| elem.produce())
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
