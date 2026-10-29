#![allow(incomplete_features)]
#![feature(portable_simd, with_negative_coherence)]

use std::{
    error::Error,
    fs::File,
    io::BufWriter,
    sync::{Arc, Barrier, RwLock},
    thread,
};

use lib::{
    Simulation,
    core::{ImageType, Synchronizer, stat::Stat},
    estimator::{Estimator, classical::ClassicalEstimator, quantum::QuantumEstimator},
    output::EstimatorsOutputOption,
};

use crate::{
    core::Unimplemented,
    estimator::{
        EstimatorsError,
        quantum::{
            potential_energy::PotentialEnergy, primitive_kinetic_energy::PrimitiveKineticEnergy,
            virial_kinetic_energy::VirialKineticEnergy,
        },
    },
    output::value::DefaultStream,
    potential::{exchange::distinguishable::DistinguishableExchangePotential, physical::harmonic::Harmonic},
    propagator::VelocityVerlet,
    sync_ops::mutex::MutexSyncOp,
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

fn main() {
    const DIM: usize = 3;
    let steps = 1_000;
    let step_size = 0.001;
    let atoms = 3;
    let images = 32;
    let mass = 1.0;
    let temperature = 1.0;
    let spring_constant = 1.0;
    let system_synchronizer = &Synchronizer::new(RwLock::new(0.0), images * atoms);
    let mut image_synchronizers = &*Vec::from_fn(images, |_| Synchronizer::new(RwLock::new(0.0), 1));
    let estimators_barrier = &Barrier::new(images);
    let system_adder = MutexSyncOp::<f64>::new(images - 1);
    let mut image_adders = Vec::from_fn(images, |_| MutexSyncOp::<f64>::new(0)).into_iter();
    let estimators_adder = MutexSyncOp::new(images - 1);
    let system_multiplier = MutexSyncOp::<f64>::new(images - 1);
    let mut image_multipliers = Vec::from_fn(images, |_| MutexSyncOp::<f64>::new(0)).into_iter();
    let propagators_barrier = Arc::new(Barrier::new(images));
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(images - 1);

        handles.push(s.spawn::<_, Result<_, Box<dyn Error + Send + Sync>>>(|| {
            let mut leading_image_simulation =
                Simulation {
                    images,
                    image: ImageType::Leading,
                    group: 0,
                    system_synchronizer: system_synchronizer,
                    image_synchronizer: image_synchronizers.split_off_first().unwrap(),
                    type_synchronizer: &Synchronizer::new(RwLock::new(0.0), 1),
                    estimators_barrier,
                    system_adder,
                    image_adder: image_adders.next().unwrap(),
                    estimators_adder,
                    system_multiplier,
                    image_multiplier: image_multipliers.next().unwrap(),
                    physical_potential: Harmonic::new(images, spring_constant, MutexSyncOp::new(0)),
                    exchange_potential: Stat::Distinguishable::<_, Unimplemented>(
                        DistinguishableExchangePotential::new(images, mass, temperature, MutexSyncOp::new(0)),
                    ),
                    thermostat: Langevin::new(mass, temperature, 1.0, rand::rng()),
                    propagator: VelocityVerlet::new(step_size, mass, propagators_barrier.clone()),
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
                &mut DefaultStream<BufWriter<File>>,
            > = EstimatorsOutputOption::Shared {
                quantum_estimators: &mut [
                    Estimator::Value(&mut PotentialEnergy),
                    Estimator::Value(&mut PrimitiveKineticEnergy::new(DIM, images, atoms, temperature)),
                    Estimator::Value(&mut VirialKineticEnergy::new(images)),
                ],
                classical_estimators: &mut [],
                stream: &mut DefaultStream::new(BufWriter::new(File::create_new("output.dat").unwrap()), steps, 4),
            };

            for step in 0..steps {
                leading_image_simulation.step_output::<_, _, _, f64, f64, _, Box<dyn Error + Send + Sync>, _>(
                    step,
                    estimators_output.as_deref_mut(),
                    Option::<Unimplemented>::None,
                    None,
                    None,
                    None,
                    todo!(),
                    todo!(),
                    todo!(),
                    todo!(),
                )?;
            }

            Ok(())
        }));
    });
}
