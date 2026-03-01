#![feature(new_range_api, ptr_metadata)]
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
    core::{AtomType, CommError, Factory, FullFactory, GroupImageHandle, GroupTypeHandle},
    observable::{
        debug::{InnerDebugObservable, LeadingDebugObservable, MainDebugObservable, TrailingDebugObservable},
        quantum::{InnerQuantumObservable, LeadingQuantumObservable, MainQuantumObservable, TrailingQuantumObservable},
    },
    output::{ObservableOutput, ObservableOutputOption, ObservableStreamOption, VectorsOutput},
    potential::{
        exchange::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential},
        physical::PhysicalPotential,
    },
    propagator::{InnerPropagator, LeadingPropagator, TrailingPropagator},
    stat::{Bosonic, Distinguishable, Stat},
    stride::StridesMut,
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
mod stride;
pub mod sync_ops;
pub mod thermostat;
pub mod vector;

fn run<
    const N: usize,
    T: Clone + Default + From<f32> + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Display + Send + Sync,
    V: Vector<N, Element = T> + Display + Send,
    AdderReciever: SyncAddRecv<T> + ?Sized,
    AdderSender: SyncAddSend<T> + Send + ?Sized,
    MultiplierReciever: SyncMulRecv<T> + ?Sized,
    MultiplierSender: SyncMulSend<T> + Send + ?Sized,
    CoordOut: VectorsOutput<N, T, V> + ?Sized,
    QObsMain: MainQuantumObservable<T, V, AdderReciever, MultiplierReciever, Output = Output> + Send + ?Sized,
    QObsLeading: LeadingQuantumObservable<T, V, AdderSender, MultiplierSender, DistLeading, BosonLeading, Output = Output>
        + Send
        + ?Sized,
    QObsInner: InnerQuantumObservable<T, V, AdderSender, MultiplierSender, DistInner, BosonInner, Output = Output> + Send + ?Sized,
    QObsTrailing: TrailingQuantumObservable<T, V, AdderSender, MultiplierSender, DistTrailing, BosonTrailing, Output = Output>
        + Send
        + ?Sized,
    DObsMain: MainDebugObservable<T, V, AdderReciever, MultiplierReciever, Output = Output> + ?Sized,
    DObsLeading: LeadingDebugObservable<T, V, AdderSender, MultiplierSender, DistLeading, BosonLeading, Output = Output>
        + Send
        + ?Sized,
    DObsInner: InnerDebugObservable<T, V, AdderSender, MultiplierSender, DistInner, BosonInner, Output = Output> + Send + ?Sized,
    DObsTrailing: TrailingDebugObservable<T, V, AdderSender, MultiplierSender, DistTrailing, BosonTrailing, Output = Output>
        + Send
        + ?Sized,
    ObsOut: ObservableOutput<Output> + ?Sized,
    PropLeading: LeadingPropagator<T, V, Phys, DistLeading, BosonLeading, Therm> + Send + ?Sized,
    PropInner: InnerPropagator<T, V, Phys, DistInner, BosonInner, Therm> + Send + ?Sized,
    PropTrailing: TrailingPropagator<T, V, Phys, DistTrailing, BosonTrailing, Therm> + Send + ?Sized,
    Phys: PhysicalPotential<T, V> + Send + ?Sized,
    DistLeading: LeadingExchangePotential<T, V> + Distinguishable + Send + ?Sized,
    DistInner: InnerExchangePotential<T, V> + Distinguishable + Send + ?Sized,
    DistTrailing: TrailingExchangePotential<T, V> + Distinguishable + Send + ?Sized,
    BosonLeading: LeadingExchangePotential<T, V> + Bosonic + Send + ?Sized,
    BosonInner: InnerExchangePotential<T, V> + Bosonic + Send + ?Sized,
    BosonTrailing: TrailingExchangePotential<T, V> + Bosonic + Send + ?Sized,
    Therm: Thermostat<T, V> + Send + ?Sized,
    Output,
    Err: From<CommError>
        + From<AdderReciever::Error>
        + From<AdderSender::Error>
        + From<CoordOut::Error>
        + From<ObsOut::Error>
        + From<PropLeading::Error>
        + From<PropInner::Error>
        + From<PropTrailing::Error>
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
    inner_images: usize,
    atom_types: &[AtomType<T>],
    groups: usize,
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
    mut positions_out: Option<
        &mut (impl Iterator<Item: DerefMut<Target = CoordOut>> + ExactSizeIterator + DoubleEndedIterator + ?Sized),
    >,
    mut momenta_out: Option<
        &mut (impl Iterator<Item: DerefMut<Target = CoordOut>> + ExactSizeIterator + DoubleEndedIterator + ?Sized),
    >,
    mut physical_forces_out: Option<
        &mut (impl Iterator<Item: DerefMut<Target = CoordOut>> + ExactSizeIterator + DoubleEndedIterator + ?Sized),
    >,
    mut exchange_forces_out: Option<
        &mut (impl Iterator<Item: DerefMut<Target = CoordOut>> + ExactSizeIterator + DoubleEndedIterator + ?Sized),
    >,
    observables: ObservableOutputOption<
        &mut [impl for<'a> FullFactory<
            'a,
            T,
            Main = &'a mut QObsMain,
            Leading = &'a mut QObsLeading,
            Inner = &'a mut QObsInner,
            Trailing = &'a mut QObsTrailing,
        >],
        &mut [impl for<'a> FullFactory<
            'a,
            T,
            Main = &'a mut DObsMain,
            Leading = &'a mut DObsLeading,
            Inner = &'a mut DObsInner,
            Trailing = &'a mut DObsTrailing,
        >],
        &mut ObsOut,
    >,
    propagators: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Leading = &'a mut PropLeading,
        Inner = &'a mut PropInner,
        Trailing = &'a mut PropTrailing,
    > + ?Sized
         ),
    physical_potentials: &mut (
             impl for<'a> Factory<'a, T, Leading = &'a mut Phys, Inner = &'a mut Phys, Trailing = &'a mut Phys> + ?Sized
         ),
    exchange_potentials: &mut (
             impl for<'a> Factory<
        'a,
        T,
        Leading = Stat<&'a mut DistLeading, &'a mut BosonLeading>,
        Inner = Stat<&'a mut DistInner, &'a mut BosonInner>,
        Trailing = Stat<&'a mut DistTrailing, &'a mut BosonTrailing>,
    > + ?Sized
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
    let run_step_leading_group = |step: usize,
                                  barrier: &Barrier,
                                  shared_value: &RwLock<T>,
                                  atom_type: &AtomType<T>,
                                  group: usize,
                                  adder: &mut AdderSender,
                                  multiplier: &mut MultiplierSender,
                                  mut quantum_observables: Option<&mut [&mut QObsLeading]>,
                                  mut debug_observables: Option<&mut [&mut DObsLeading]>,
                                  propagator: &mut PropLeading,
                                  physical_potential: &mut Phys,
                                  mut exchange_potential: Stat<&mut DistLeading, &mut BosonLeading>,
                                  thermostat: &mut Therm,
                                  positions: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
                                  momenta: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
                                  physical_forces: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
                                  exchange_forces: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>|
     -> Result<(), Err> {
        let (group_physical_potential_energy, group_exchange_potential_energy) = propagator.propagate(
            step,
            physical_potential,
            exchange_potential.as_deref_mut(),
            thermostat,
            &mut *positions.write(),
            &mut *momenta.write(),
            &mut *physical_forces.write(),
            &mut *exchange_forces.write(),
        )?;

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

        if let Some(observables) = quantum_observables.as_deref_mut() {
            for observable in observables {
                observable.calculate(
                    exchange_potential.as_deref(),
                    adder,
                    multiplier,
                    physical_potential_energy.clone(),
                    exchange_potential_energy.clone(),
                    &*positions.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*physical_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*exchange_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                )?;
                barrier.wait();
            }
        }

        if let Some(observables) = debug_observables.as_deref_mut() {
            for observable in observables {
                observable.calculate(
                    exchange_potential.as_deref(),
                    adder,
                    multiplier,
                    physical_potential_energy.clone(),
                    exchange_potential_energy.clone(),
                    kinetic_energy.clone(),
                    &*positions.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*momenta.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*physical_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*exchange_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                )?;
                barrier.wait();
            }
        }

        Ok(())
    };

    let run_step_inner_group = |step: usize,
                                barrier: &Barrier,
                                shared_value: &RwLock<T>,
                                image: usize,
                                atom_type: &AtomType<T>,
                                group: usize,
                                adder: &mut AdderSender,
                                multiplier: &mut MultiplierSender,
                                mut quantum_observables: Option<&mut [&mut QObsInner]>,
                                mut debug_observables: Option<&mut [&mut DObsInner]>,
                                propagator: &mut PropInner,
                                physical_potential: &mut Phys,
                                mut exchange_potential: Stat<&mut DistInner, &mut BosonInner>,
                                thermostat: &mut Therm,
                                positions: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
                                momenta: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
                                physical_forces: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
                                exchange_forces: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>|
     -> Result<(), Err> {
        let (group_physical_potential_energy, group_exchange_potential_energy) = propagator.propagate(
            step,
            physical_potential,
            exchange_potential.as_deref_mut(),
            thermostat,
            &mut *positions.write(),
            &mut *momenta.write(),
            &mut *physical_forces.write(),
            &mut *exchange_forces.write(),
        )?;

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

        if let Some(observables) = quantum_observables.as_deref_mut() {
            for observable in observables {
                observable.calculate(
                    exchange_potential.as_deref(),
                    adder,
                    multiplier,
                    physical_potential_energy.clone(),
                    exchange_potential_energy.clone(),
                    &*positions.read_whole().map_err(|_| CommError::Inner { image, group })?,
                    &*physical_forces
                        .read_whole()
                        .map_err(|_| CommError::Inner { image, group })?,
                    &*exchange_forces
                        .read_whole()
                        .map_err(|_| CommError::Inner { image, group })?,
                )?;
                barrier.wait();
            }
        }

        if let Some(observables) = debug_observables.as_deref_mut() {
            for observable in observables {
                observable.calculate(
                    exchange_potential.as_deref(),
                    adder,
                    multiplier,
                    physical_potential_energy.clone(),
                    exchange_potential_energy.clone(),
                    kinetic_energy.clone(),
                    &*positions.read_whole().map_err(|_| CommError::Inner { image, group })?,
                    &*momenta.read_whole().map_err(|_| CommError::Inner { image, group })?,
                    &*physical_forces
                        .read_whole()
                        .map_err(|_| CommError::Inner { image, group })?,
                    &*exchange_forces
                        .read_whole()
                        .map_err(|_| CommError::Inner { image, group })?,
                )?;
                barrier.wait();
            }
        }

        Ok(())
    };

    let run_step_trailing_group = |step: usize,
                                   barrier: &Barrier,
                                   shared_value: &RwLock<T>,
                                   atom_type: &AtomType<T>,
                                   group: usize,
                                   adder: &mut AdderSender,
                                   multiplier: &mut MultiplierSender,
                                   mut quantum_observables: Option<&mut [&mut QObsTrailing]>,
                                   mut debug_observables: Option<&mut [&mut DObsTrailing]>,
                                   propagator: &mut PropTrailing,
                                   physical_potential: &mut Phys,
                                   mut exchange_potential: Stat<&mut DistTrailing, &mut BosonTrailing>,
                                   thermostat: &mut Therm,
                                   positions: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
                                   momenta: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
                                   physical_forces: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>,
                                   exchange_forces: &mut ElementRwLock<GroupImageHandle<GroupTypeHandle<V>>>|
     -> Result<(), Err> {
        let (group_physical_potential_energy, group_exchange_potential_energy) = propagator.propagate(
            step,
            physical_potential,
            exchange_potential.as_deref_mut(),
            thermostat,
            &mut *positions.write(),
            &mut *momenta.write(),
            &mut *physical_forces.write(),
            &mut *exchange_forces.write(),
        )?;

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

        if let Some(observables) = quantum_observables.as_deref_mut() {
            for observable in observables {
                observable.calculate(
                    exchange_potential.as_deref(),
                    adder,
                    multiplier,
                    physical_potential_energy.clone(),
                    exchange_potential_energy.clone(),
                    &*positions.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*physical_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*exchange_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                )?;
                barrier.wait();
            }
        }

        if let Some(observables) = debug_observables.as_deref_mut() {
            for observable in observables {
                observable.calculate(
                    exchange_potential.as_deref(),
                    adder,
                    multiplier,
                    physical_potential_energy.clone(),
                    exchange_potential_energy.clone(),
                    kinetic_energy.clone(),
                    &*positions.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*momenta.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*physical_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                    &*exchange_forces.read_whole().map_err(|_| CommError::Leading { group })?,
                )?;
                barrier.wait();
            }
        }
        Ok(())
    };

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

    let (leading_positions_out, inner_positions_out_iter, trailing_positions_out) =
        if let Some(iter) = positions_out.as_deref_mut() {
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
    let (leading_momenta_out, inner_momenta_out_iter, trailing_momenta_out) =
        if let Some(iter) = momenta_out.as_deref_mut() {
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
    let (leading_physical_forces_out, inner_physical_forces_out_iter, trailing_physical_forces_out) =
        if let Some(iter) = physical_forces_out.as_deref_mut() {
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
    let (leading_exchange_forces_out, inner_exchange_forces_out_iter, trailing_exchange_forces_out) =
        if let Some(iter) = exchange_forces_out.as_deref_mut() {
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

    let (quantum_observables, debug_observables, mut streams) = observables.split();

    let (
        n_quantum_observables,
        mut main_quantum_observables,
        mut leading_quantum_observables,
        mut inner_quantum_observables,
        mut trailing_quantum_observables,
    ) = if let Some(observables) = quantum_observables {
        let n_observables = observables.len();
        let mut uninit_main_observables = Box::new_uninit_slice(n_observables);
        let mut uninit_leading_observables = Box::new_uninit_slice(groups * n_observables);
        let mut uninit_inner_observables = Box::new_uninit_slice(groups * inner_images * n_observables);
        let mut uninit_trailing_observables = Box::new_uninit_slice(groups * n_observables);
        for zip_items!(
            observable,
            uninit_main_observable,
            uninit_leading_observables,
            mut uninit_inner_observables,
            uninit_trailing_observables
        ) in zip_iterators!(
            observables.iter_mut(),
            uninit_main_observables.iter_mut(),
            StridesMut::from_slice(&mut *uninit_leading_observables, n_observables),
            StridesMut::from_slice(&mut *uninit_inner_observables, n_observables),
            StridesMut::from_slice(&mut *uninit_trailing_observables, n_observables)
        ) {
            let (main_observable, leading_observables, inner_observables_iter, trailing_observables) =
                observable.produce(inner_images, atom_types);

            assert_eq!(leading_observables.len(), groups);
            assert_eq!(inner_observables_iter.len(), inner_images);
            assert_eq!(trailing_observables.len(), groups);

            uninit_main_observable.write(main_observable);

            for zip_items!(
                uninit_leading_observable,
                uninit_trailing_observable,
                leading_observable,
                trailing_observable
            ) in zip_iterators!(
                uninit_leading_observables,
                uninit_trailing_observables,
                leading_observables,
                trailing_observables
            ) {
                uninit_leading_observable.write(leading_observable);
                uninit_trailing_observable.write(trailing_observable);
            }

            for inner_observables in inner_observables_iter {
                assert_eq!(inner_observables.len(), groups);

                for (uninit_inner_observable, inner_observable) in
                    uninit_inner_observables.by_ref().take(groups).zip(inner_observables)
                {
                    uninit_inner_observable.write(inner_observable);
                }
            }
        }
        // SAFETY: Initialized all elements above.
        unsafe {
            (
                n_observables,
                Some(uninit_main_observables.assume_init()),
                Some(uninit_leading_observables.assume_init()),
                Some(uninit_inner_observables.assume_init()),
                Some(uninit_trailing_observables.assume_init()),
            )
        }
    } else {
        (0, None, None, None, None)
    };

    let (
        n_debug_observables,
        mut main_debug_observables,
        mut leading_debug_observables,
        mut inner_debug_observables,
        mut trailing_debug_observables,
    ) = if let Some(observables) = debug_observables {
        let n_observables = observables.len();
        let mut uninit_main_observables = Box::new_uninit_slice(n_observables);
        let mut uninit_leading_observables = Box::new_uninit_slice(groups * n_observables);
        let mut uninit_inner_observables = Box::new_uninit_slice(groups * inner_images * n_observables);
        let mut uninit_trailing_observables = Box::new_uninit_slice(groups * n_observables);
        for zip_items!(
            observable,
            uninit_main_observable,
            uninit_leading_observables,
            mut uninit_inner_observables,
            uninit_trailing_observables
        ) in zip_iterators!(
            observables.iter_mut(),
            uninit_main_observables.iter_mut(),
            StridesMut::from_slice(&mut *uninit_leading_observables, n_observables),
            StridesMut::from_slice(&mut *uninit_inner_observables, n_observables),
            StridesMut::from_slice(&mut *uninit_trailing_observables, n_observables)
        ) {
            let (main_observable, leading_observables, inner_observables_iter, trailing_observables) =
                observable.produce(inner_images, atom_types);

            assert_eq!(leading_observables.len(), groups);
            assert_eq!(inner_observables_iter.len(), inner_images);
            assert_eq!(trailing_observables.len(), groups);

            uninit_main_observable.write(main_observable);

            for zip_items!(
                uninit_leading_observable,
                uninit_trailing_observable,
                leading_observable,
                trailing_observable
            ) in zip_iterators!(
                uninit_leading_observables,
                uninit_trailing_observables,
                leading_observables,
                trailing_observables
            ) {
                uninit_leading_observable.write(leading_observable);
                uninit_trailing_observable.write(trailing_observable);
            }

            for inner_observables in inner_observables_iter {
                assert_eq!(inner_observables.len(), groups);

                for (uninit_inner_observable, inner_observable) in
                    uninit_inner_observables.by_ref().take(groups).zip(inner_observables)
                {
                    uninit_inner_observable.write(inner_observable);
                }
            }
        }
        // SAFETY: Initialized all elements above.
        unsafe {
            (
                n_observables,
                Some(uninit_main_observables.assume_init()),
                Some(uninit_leading_observables.assume_init()),
                Some(uninit_inner_observables.assume_init()),
                Some(uninit_trailing_observables.assume_init()),
            )
        }
    } else {
        (0, None, None, None, None)
    };

    let barrier = Barrier::new(inner_images + 3);
    let shared_value = RwLock::new(T::default());

    let barrier = &barrier;
    let shared_value = &shared_value;

    let (main_adder, leading_adders, inner_adders_iter, trailing_adders) = adders.produce(inner_images, atom_types);
    assert_eq!(leading_adders.len(), groups);
    assert_eq!(inner_adders_iter.len(), inner_images);
    assert_eq!(trailing_adders.len(), groups);

    let (main_multiplier, leading_multipliers, inner_multipliers_iter, trailing_multipliers) =
        multipliers.produce(inner_images, atom_types);
    assert_eq!(leading_multipliers.len(), groups);
    assert_eq!(inner_multipliers_iter.len(), inner_images);
    assert_eq!(trailing_multipliers.len(), groups);

    let (leading_propagators, inner_propagators_iter, trailing_propagators) =
        propagators.produce(inner_images, atom_types);
    assert_eq!(leading_propagators.len(), groups);
    assert_eq!(inner_propagators_iter.len(), inner_images);
    assert_eq!(trailing_propagators.len(), groups);

    let (leading_physical_potentials, inner_physical_potentials_iter, trailing_physical_potentials) =
        physical_potentials.produce(inner_images, atom_types);
    assert_eq!(leading_physical_potentials.len(), groups);
    assert_eq!(inner_physical_potentials_iter.len(), inner_images);
    assert_eq!(trailing_physical_potentials.len(), groups);

    let (leading_exchange_potentials, inner_exchange_potentials_iter, trailing_exchange_potentials) =
        exchange_potentials.produce(inner_images, atom_types);
    assert_eq!(leading_exchange_potentials.len(), groups);
    assert_eq!(inner_exchange_potentials_iter.len(), inner_images);
    assert_eq!(trailing_exchange_potentials.len(), groups);

    let (leading_thermostats, inner_thermostats_iter, trailing_thermostats) =
        thermostats.produce(inner_images, atom_types);
    assert_eq!(leading_thermostats.len(), groups);
    assert_eq!(inner_thermostats_iter.len(), inner_images);
    assert_eq!(trailing_thermostats.len(), groups);

    let (leading_positions, inner_positions_iter, trailing_positions) = positions.produce(inner_images, atom_types);
    assert_eq!(leading_positions.len(), groups);
    assert_eq!(inner_positions_iter.len(), inner_images);
    assert_eq!(trailing_positions.len(), groups);

    let (leading_momenta, inner_momenta_iter, trailing_momenta) = momenta.produce(inner_images, atom_types);
    assert_eq!(leading_momenta.len(), groups);
    assert_eq!(inner_momenta_iter.len(), inner_images);
    assert_eq!(trailing_momenta.len(), groups);

    let (leading_physical_forces, inner_physical_forces_iter, trailing_physical_forces) =
        physical_forces.produce(inner_images, atom_types);
    assert_eq!(leading_physical_forces.len(), groups);
    assert_eq!(inner_physical_forces_iter.len(), inner_images);
    assert_eq!(trailing_physical_forces.len(), groups);

    let (leading_exchange_forces, inner_exchange_forces_iter, trailing_exchange_forces) =
        exchange_forces.produce(inner_images, atom_types);
    assert_eq!(leading_exchange_forces.len(), groups);
    assert_eq!(inner_exchange_forces_iter.len(), inner_images);
    assert_eq!(trailing_exchange_forces.len(), groups);

    thread::scope(|s| {
        let mut atom_types_iter = atom_types.iter();
        let mut leading_quantum_observables_iter = leading_quantum_observables
            .as_deref_mut()
            .map(|observables| observables.chunks_exact_mut(n_quantum_observables));
        let mut leading_debug_observables_iter = leading_debug_observables
            .as_deref_mut()
            .map(|observables| observables.chunks_exact_mut(n_debug_observables));
        for zip_items!(
            (atom_type, group),
            adder,
            multiplier,
            mut quantum_observables,
            mut debug_observables,
            propagator,
            physical_potential,
            mut exchange_potential,
            thermostat,
            mut positions,
            mut momenta,
            mut physical_forces,
            mut exchange_forces
        ) in zip_iterators!(
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
                match &mut leading_quantum_observables_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut leading_debug_observables_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            leading_propagators,
            leading_physical_potentials,
            leading_exchange_potentials,
            leading_thermostats,
            leading_positions,
            leading_momenta,
            leading_physical_forces,
            leading_exchange_forces
        ) {
            s.spawn::<_, Result<_, Err>>(move || {
                for step in 0..steps {
                    run_step_leading_group(
                        step,
                        barrier,
                        shared_value,
                        atom_type,
                        group,
                        adder,
                        multiplier,
                        quantum_observables.as_deref_mut(),
                        debug_observables.as_deref_mut(),
                        propagator,
                        physical_potential,
                        exchange_potential.as_deref_mut(),
                        thermostat,
                        &mut positions,
                        &mut momenta,
                        &mut physical_forces,
                        &mut exchange_forces,
                    )?;
                }
                Ok(())
            });
        }

        let mut inner_quantum_observables_iter = inner_quantum_observables
            .as_deref_mut()
            .map(|observables| observables.chunks_exact_mut(groups * n_quantum_observables));
        let mut inner_debug_observables_iter = inner_debug_observables
            .as_deref_mut()
            .map(|observables| observables.chunks_exact_mut(groups * n_debug_observables));
        for zip_items!(
            image,
            inner_adders,
            inner_multipliers,
            inner_quantum_observables,
            inner_debug_observables,
            inner_propagators,
            inner_physical_potentials,
            inner_exchange_potentials,
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
                match &mut inner_quantum_observables_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut inner_debug_observables_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            inner_propagators_iter,
            inner_physical_potentials_iter,
            inner_exchange_potentials_iter,
            inner_thermostats_iter,
            inner_positions_iter,
            inner_momenta_iter,
            inner_physical_forces_iter,
            inner_exchange_forces_iter
        ) {
            let mut atom_types_iter = atom_types.iter();
            let mut quantum_observables_iter =
                inner_quantum_observables.map(|observables| observables.chunks_exact_mut(n_quantum_observables));
            let mut debug_observables_iter =
                inner_debug_observables.map(|observables| observables.chunks_exact_mut(n_debug_observables));
            for zip_items!(
                (atom_type, group),
                adder,
                multiplier,
                mut quantum_observables,
                mut debug_observables,
                propagator,
                physical_potential,
                mut exchange_potential,
                thermostat,
                mut positions,
                mut momenta,
                mut physical_forces,
                mut exchange_forces
            ) in zip_iterators!(
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
                    match &mut quantum_observables_iter {
                        Some(iter) => iter.next().map(Some),
                        None => Some(None),
                    }
                }),
                iter::from_fn(|| {
                    match &mut debug_observables_iter {
                        Some(iter) => iter.next().map(Some),
                        None => Some(None),
                    }
                }),
                inner_propagators,
                inner_physical_potentials,
                inner_exchange_potentials,
                inner_thermostats,
                inner_positions,
                inner_momenta,
                inner_physical_forces,
                inner_exchange_forces
            ) {
                s.spawn::<_, Result<_, Err>>(move || {
                    for step in 0..steps {
                        run_step_inner_group(
                            step,
                            barrier,
                            shared_value,
                            image,
                            atom_type,
                            group,
                            adder,
                            multiplier,
                            quantum_observables.as_deref_mut(),
                            debug_observables.as_deref_mut(),
                            propagator,
                            physical_potential,
                            exchange_potential.as_deref_mut(),
                            thermostat,
                            &mut positions,
                            &mut momenta,
                            &mut physical_forces,
                            &mut exchange_forces,
                        )?;
                    }
                    Ok(())
                });
            }
        }

        let mut atom_types_iter = atom_types.iter();
        let mut trailing_quantum_observables_iter = trailing_quantum_observables
            .as_deref_mut()
            .map(|observables| observables.chunks_exact_mut(n_quantum_observables));
        let mut trailing_debug_observables_iter = trailing_debug_observables
            .as_deref_mut()
            .map(|observables| observables.chunks_exact_mut(n_debug_observables));
        for zip_items!(
            (atom_type, group),
            adder,
            multiplier,
            mut quantum_observables,
            mut debug_observables,
            propagator,
            physical_potential,
            mut exchange_potential,
            thermostat,
            mut positions,
            mut momenta,
            mut physical_forces,
            mut exchange_forces
        ) in zip_iterators!(
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
                match &mut trailing_quantum_observables_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            iter::from_fn(|| {
                match &mut trailing_debug_observables_iter {
                    Some(iter) => iter.next().map(Some),
                    None => Some(None),
                }
            }),
            trailing_propagators,
            trailing_physical_potentials,
            trailing_exchange_potentials,
            trailing_thermostats,
            trailing_positions,
            trailing_momenta,
            trailing_physical_forces,
            trailing_exchange_forces
        ) {
            s.spawn::<_, Result<_, Err>>(move || {
                for step in 0..steps {
                    run_step_trailing_group(
                        step,
                        barrier,
                        shared_value,
                        atom_type,
                        group,
                        adder,
                        multiplier,
                        quantum_observables.as_deref_mut(),
                        debug_observables.as_deref_mut(),
                        propagator,
                        physical_potential,
                        exchange_potential.as_deref_mut(),
                        thermostat,
                        &mut positions,
                        &mut momenta,
                        &mut physical_forces,
                        &mut exchange_forces,
                    )?;
                }
                Ok(())
            });
        }

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

            match (
                streams.as_deref_mut(),
                main_quantum_observables.as_deref_mut(),
                main_debug_observables.as_deref_mut(),
            ) {
                (ObservableStreamOption::None, None, None) => {}
                (ObservableStreamOption::One(stream), Some(observables), None) => {
                    stream.write_step(step)?;
                    for observable in observables {
                        stream.write_observable(observable.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    stream.new_line()?;
                }
                (ObservableStreamOption::One(stream), None, Some(observables)) => {
                    stream.write_step(step)?;
                    for observable in observables {
                        stream.write_observable(observable.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    stream.new_line()?;
                }
                (ObservableStreamOption::Shared(stream), Some(quantum_observables), Some(debug_observables)) => {
                    stream.write_step(step)?;
                    for observable in quantum_observables {
                        stream.write_observable(observable.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    for observable in debug_observables {
                        stream.write_observable(observable.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    stream.new_line()?;
                }
                (
                    ObservableStreamOption::Separate {
                        quantum: quantum_stream,
                        debug: debug_stream,
                    },
                    Some(quantum_observables),
                    Some(debug_observables),
                ) => {
                    quantum_stream.write_step(step)?;
                    for observable in quantum_observables {
                        quantum_stream.write_observable(observable.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    quantum_stream.new_line()?;

                    debug_stream.write_step(step)?;
                    for observable in debug_observables {
                        debug_stream.write_observable(observable.calculate(main_adder, main_multiplier)?)?;
                        barrier.wait();
                    }
                    debug_stream.new_line()?;
                }
                _ => panic!("Nonsensical output combination"),
            }
            barrier.wait();
        }

        todo!()
    })
}
