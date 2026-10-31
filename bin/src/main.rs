#![allow(incomplete_features)]
#![feature(portable_simd, with_negative_coherence, specialization)]

use std::{
    error::Error,
    fs::File,
    io::BufWriter,
    num::NonZeroUsize,
    panic,
    sync::{Arc, Barrier, RwLock},
    thread,
};

use arc_rw_lock::UniqueArcSliceRwLock;
use lib::{
    Simulation,
    core::{
        AtomType, AtomTypeCharacteristics, GroupRwLockInTypeInImageInSystem, ImageType, MapInWhole, Synchronizer,
        stat::Stat,
    },
    estimator::{Estimator, classical::ClassicalEstimator, quantum::QuantumEstimator},
    output::EstimatorsOutputOption,
    zip_items, zip_iterators,
};

use crate::{
    core::{Dummy, Unimplemented},
    estimator::quantum::{
        potential_energy::PotentialEnergy, primitive_kinetic_energy::PrimitiveKineticEnergy,
        virial_kinetic_energy::VirialKineticEnergy,
    },
    output::{value::DefaultStream, xyz::XYZFileStream},
    potential::{exchange::distinguishable::DistinguishableExchangePotential, physical::harmonic::Harmonic},
    propagator::VelocityVerlet,
    sync_ops::mutex,
    thermostat::langevin::Langevin,
    vector::SimdVector,
};

pub mod core;
pub mod estimator;
pub mod output;
pub mod potential;
pub mod propagator;
pub mod sync_ops;
pub mod thermostat;
pub mod vector;

type SimulationError = Box<dyn Error + Send + Sync>;

fn main() -> Result<(), SimulationError> {
    const DIM: usize = 3;
    let steps = 10;
    let step_size = 0.001;
    let atoms = 3;
    let images = 2;
    let characteristics = AtomTypeCharacteristics {
        mass: 1.0,
        charge: 0.0,
        name: "H".into(),
    };
    let temperature = 1.0;
    let spring_constant = 1.0;
    let system_synchronizer = &Synchronizer::new(RwLock::new(0.0), images);
    let estimators_barrier = &Barrier::new(images);
    let (system_adder_receiver, system_adder_senders) = mutex::mutex_sync_op::<f64>(images - 1);
    let (estimators_adder_receiver, estimators_adder_senders) = mutex::mutex_sync_op::<f64>(images - 1);
    let (system_multiplier_receiver, system_multiplier_senders) = mutex::mutex_sync_op::<f64>(images - 1);
    let propagators_barrier = Arc::new(Barrier::new(images));

    let yield_coordinates = || {
        let mut coordinates_locks = Box::new_uninit_slice(images);
        let mut coordinates_types = Box::new_uninit_slice(images);
        for zip_items!(coordinates_lock, coordinates_type, image) in zip_iterators!(
            coordinates_locks.iter_mut(),
            coordinates_types.iter_mut(),
            // SAFETY: A zeroed-out `SimdVector<f64, _>` is still valid.
            unsafe {
                UniqueArcSliceRwLock::<SimdVector<f64, DIM>>::new_zeroed_slice(images * atoms)
                    .0
                    .assume_init()
            }
            .chunks_exact_mut(atoms)
        ) {
            let (mut rw_lock, reader_lock) = UniqueArcSliceRwLock::new_uninit_slice(1);
            rw_lock.write().get_mut(0).unwrap().write(image);
            // SAFETY: Initialized the only element above.
            coordinates_lock.write(unsafe { rw_lock.assume_init() }.iter_mut().next().unwrap());
            coordinates_type.write(AtomType {
                characteristics: &characteristics,
                // SAFETY: Initialized the only element above.
                lock: unsafe { reader_lock.assume_init() },
            });
        }
        // SAFETY: Initialized all elements above.
        (unsafe { coordinates_locks.assume_init() }.into_iter(), unsafe {
            coordinates_types.assume_init()
        })
    };

    let (mut positions_locks, positions_types) = yield_coordinates();
    let positions_types = &*positions_types;

    let (mut momenta_locks, momenta_types) = yield_coordinates();
    let momenta_types = &*momenta_types;

    let (mut physical_forces_locks, physical_forces_types) = yield_coordinates();
    let physical_forces_types = &*physical_forces_types;

    let (mut exchange_forces_locks, exchange_forces_types) = yield_coordinates();
    let exchange_forces_types = &*exchange_forces_types;

    thread::scope(|s| {
        let mut handles = Vec::with_capacity(images - 1);

        let mut leading_positions_lock = positions_locks.next().unwrap();
        let mut leading_momenta_lock = momenta_locks.next().unwrap();
        let mut leading_physical_forces_lock = physical_forces_locks.next().unwrap();
        let mut leading_exchange_forces_lock = exchange_forces_locks.next().unwrap();

        for zip_items!(
            image,
            system_adder,
            estimators_adder,
            system_multiplier,
            mut positions_lock,
            mut momenta_lock,
            mut physical_forces_lock,
            mut exchange_forces_lock,
        ) in zip_iterators!(
            1..images,
            system_adder_senders,
            estimators_adder_senders,
            system_multiplier_senders,
            positions_locks,
            momenta_locks,
            physical_forces_locks,
            exchange_forces_locks
        ) {
            let propagators_barrier = propagators_barrier.clone();
            let image = NonZeroUsize::new(image).unwrap();
            handles.push(s.spawn::<_, Result<_, SimulationError>>(move || {
                let mut simulation = Simulation {
                    images,
                    image: if image.get() == images - 1 {
                        ImageType::Trailing(image)
                    } else {
                        ImageType::Inner(image)
                    },
                    group: 0,
                    system_synchronizer: system_synchronizer,
                    image_synchronizer: &Synchronizer::new(RwLock::new(0.0), 1),
                    type_synchronizer: &Synchronizer::new(RwLock::new(0.0), 1),
                    estimators_barrier,
                    system_adder,
                    image_adder: mutex::mutex_sync_op::<f64>(0).0,
                    estimators_adder,
                    system_multiplier,
                    image_multiplier: mutex::mutex_sync_op::<f64>(0).0,
                    physical_potential: Harmonic::new(images, spring_constant, mutex::mutex_sync_op::<f64>(0).0),
                    exchange_potential: Stat::Distinguishable::<_, Unimplemented>(
                        DistinguishableExchangePotential::new(
                            images,
                            characteristics.mass,
                            temperature,
                            mutex::mutex_sync_op::<f64>(0).0,
                        ),
                    ),
                    thermostat: Langevin::new(characteristics.mass, temperature, 1.0, rand::rng()),
                    propagator: VelocityVerlet::new(step_size, characteristics.mass, propagators_barrier),
                };

                let quantum_estimators: &mut [Estimator<
                    &mut dyn QuantumEstimator<_, _, _, _, _, Output = f64, Error = _>,
                    &mut dyn QuantumEstimator<_, _, _, _, _, Output = SimdVector<_, DIM>, Error = _>,
                >] = &mut [
                    Estimator::Value(&mut PotentialEnergy),
                    Estimator::Value(&mut PrimitiveKineticEnergy::new(DIM, images, atoms, temperature)),
                    Estimator::Value(&mut VirialKineticEnergy::new(images)),
                ];

                let classical_estimators: &mut [Estimator<
                    &mut dyn ClassicalEstimator<_, _, _, _, _, Output = f64, Error = _>,
                    &mut dyn ClassicalEstimator<_, _, _, _, _, Output = SimdVector<_, DIM>, Error = _>,
                >] = &mut [];

                let positions = &mut GroupRwLockInTypeInImageInSystem {
                    map: &mut positions_lock,
                    whole: unsafe {
                        MapInWhole::with_range(positions_types, image.get()..=image.get())
                            .unwrap()
                            .map(0)
                            .unwrap_unchecked()
                    },
                };
                let momenta = &mut GroupRwLockInTypeInImageInSystem {
                    map: &mut momenta_lock,
                    whole: unsafe {
                        MapInWhole::with_range(momenta_types, image.get()..=image.get())
                            .unwrap()
                            .map(0)
                            .unwrap_unchecked()
                    },
                };
                let physical_forces = &mut GroupRwLockInTypeInImageInSystem {
                    map: &mut physical_forces_lock,
                    whole: unsafe {
                        MapInWhole::with_range(physical_forces_types, image.get()..=image.get())
                            .unwrap()
                            .map(0)
                            .unwrap_unchecked()
                    },
                };
                let exchange_forces = &mut GroupRwLockInTypeInImageInSystem {
                    map: &mut exchange_forces_lock,
                    whole: unsafe {
                        MapInWhole::with_range(exchange_forces_types, image.get()..=image.get())
                            .unwrap()
                            .map(0)
                            .unwrap_unchecked()
                    },
                };

                for step in 0..steps {
                    println!("thread {} started step {}", image.get(), step);
                    simulation.step_image_leader_group::<_, f64, f64, _, SimulationError, _>(
                        step,
                        Some(quantum_estimators),
                        Some(classical_estimators),
                        positions,
                        momenta,
                        physical_forces,
                        exchange_forces,
                    )?;
                    println!("thread {} completed step {}", image.get(), step);
                }

                Ok(())
            }));
        }

        let _output_handle = s.spawn::<_, Result<_, SimulationError>>(move || {
            let mut simulation = Simulation {
                images,
                image: ImageType::Leading,
                group: 0,
                system_synchronizer: system_synchronizer,
                image_synchronizer: &Synchronizer::new(RwLock::new(0.0), 1),
                type_synchronizer: &Synchronizer::new(RwLock::new(0.0), 1),
                estimators_barrier,
                system_adder: system_adder_receiver,
                image_adder: mutex::mutex_sync_op::<f64>(0).0,
                estimators_adder: estimators_adder_receiver,
                system_multiplier: system_multiplier_receiver,
                image_multiplier: mutex::mutex_sync_op::<f64>(0).0,
                physical_potential: Harmonic::new(images, spring_constant, mutex::mutex_sync_op::<f64>(0).0),
                exchange_potential: Stat::Distinguishable::<_, Unimplemented>(DistinguishableExchangePotential::new(
                    images,
                    characteristics.mass,
                    temperature,
                    mutex::mutex_sync_op::<f64>(0).0,
                )),
                thermostat: Langevin::new(characteristics.mass, temperature, 1.0, rand::rng()),
                propagator: VelocityVerlet::new(step_size, characteristics.mass, propagators_barrier),
            };

            let mut estimators_output: EstimatorsOutputOption<
                &mut [Estimator<
                    &mut dyn QuantumEstimator<_, _, _, _, _, Output = f64, Error = _>,
                    &mut dyn QuantumEstimator<_, _, _, _, _, Output = SimdVector<_, DIM>, Error = _>,
                >],
                &mut [Estimator<
                    &mut dyn ClassicalEstimator<_, _, _, _, _, Output = f64, Error = _>,
                    &mut dyn ClassicalEstimator<_, _, _, _, _, Output = SimdVector<_, DIM>, Error = _>,
                >],
                _,
            > = EstimatorsOutputOption::Shared {
                quantum_estimators: &mut [
                    Estimator::Value(&mut PotentialEnergy),
                    Estimator::Value(&mut PrimitiveKineticEnergy::new(DIM, images, atoms, temperature)),
                    Estimator::Value(&mut VirialKineticEnergy::new(images)),
                ],
                classical_estimators: &mut [],
                stream: &mut Dummy, //DefaultStream::new(File::create_new("output.dat").unwrap(), steps, 4, 8),
            };

            let positions = &mut GroupRwLockInTypeInImageInSystem {
                map: &mut leading_positions_lock,
                whole: unsafe {
                    MapInWhole::with_range(positions_types, 0..=0)
                        .unwrap()
                        .map(0)
                        .unwrap_unchecked()
                },
            };
            let momenta = &mut GroupRwLockInTypeInImageInSystem {
                map: &mut leading_momenta_lock,
                whole: unsafe {
                    MapInWhole::with_range(momenta_types, 0..=0)
                        .unwrap()
                        .map(0)
                        .unwrap_unchecked()
                },
            };
            let physical_forces = &mut GroupRwLockInTypeInImageInSystem {
                map: &mut leading_physical_forces_lock,
                whole: unsafe {
                    MapInWhole::with_range(physical_forces_types, 0..=0)
                        .unwrap()
                        .map(0)
                        .unwrap_unchecked()
                },
            };
            let exchange_forces = &mut GroupRwLockInTypeInImageInSystem {
                map: &mut leading_exchange_forces_lock,
                whole: unsafe {
                    MapInWhole::with_range(exchange_forces_types, 0..=0)
                        .unwrap()
                        .map(0)
                        .unwrap_unchecked()
                },
            };

            for step in 0..steps {
                println!("main thread started step {}", step);
                simulation.step_output::<_, _, XYZFileStream<BufWriter<File>, _>, f64, f64, _, SimulationError, _>(
                    step,
                    estimators_output.as_deref_mut(),
                    None,
                    None,
                    None,
                    None,
                    positions,
                    momenta,
                    physical_forces,
                    exchange_forces,
                )?;
                println!("main thread completed step {}", step);
            }

            Ok(())
        });

        for zip_items!(image, handle) in zip_iterators!(1..images, handles) {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(format!("error in image {}: {}", image, err).into()),
                Err(err) => panic::resume_unwind(err),
            }
        }
        Ok(())
    })
}
