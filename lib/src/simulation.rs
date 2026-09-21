use crate::{
    core::{
        GroupRwLockInTypeInImageInSystem, ImageType, Synchronizer, Vector,
        error::CommError,
        marker::{MeaningfulOutput, ValidOutput},
        stat::{Bosonic, Distinguishable, Stat},
        sync_ops::{SyncAddReceiver, SyncAddSender},
    },
    estimator::{Estimator, classical::ClassicalEstimator, quantum::QuantumEstimator},
    output::{EstimatorsOutput, EstimatorsOutputOption, ValuesStream, VectorsStream},
    potential::{exchange::ExchangePotential, physical::PhysicalPotential},
    propagator::Propagator,
    thermostat::Thermostat,
};
use std::{
    ops::{Add, Div},
    sync::Barrier,
};

/// A trait for objects that can yield meaningful results after propagation.
/// It is an implementation detail and is implemented for '()' and any type that implements 'Clone + MeaningfulOutput'.
pub trait PropagationOutput<T> {
    /// Processes the supplied value ('self') and falliably returns a value of type 'T'.
    fn get<F, E>(self, synchronizer: &Synchronizer<T>, f: F) -> Result<T, E>
    where
        F: FnOnce() -> E;
}

impl<T: Clone> PropagationOutput<T> for () {
    /// Fetches the returned value from another thread.
    fn get<F, E>(self, synchronizer: &Synchronizer<T>, f: F) -> Result<T, E>
    where
        F: FnOnce() -> E,
    {
        synchronizer.barrier.wait();
        synchronizer
            .lock
            .read()
            .map(|guard| guard.clone())
            .map_err(|_| f())
    }
}

impl<T: Clone + MeaningfulOutput> PropagationOutput<T> for T {
    /// Sends the provided value to other threads and returns it.
    fn get<F, E>(self, synchronizer: &Synchronizer<T>, f: F) -> Result<T, E>
    where
        F: FnOnce() -> E,
    {
        synchronizer
            .lock
            .write()
            .map(|mut guard| *guard = self.clone())
            .map_err(|_| f())?;
        synchronizer.barrier.wait();
        Ok(self)
    }
}

/// An object that holds everything needed for advancing the simulation by one step.
pub struct Simulation<'a, T, SysAdd, ImAdd, EstAdd, SysMul, ImMul, Phys, Dist, Boson, Therm, Prop> {
    /// The number of images in the simulation. Often denoted in literature as 'P'.
    pub images: usize,
    /// The kind of this image.
    pub image: ImageType,
    /// The index of this group.
    pub group: usize,
    /// A synchronizer that is shared amongst all groups in all images.
    pub system_synchronizer: &'a Synchronizer<T>,
    /// A synchronizer that is shared amongst all groups in this image.
    pub image_synchronizer: &'a Synchronizer<T>,
    /// A synchronizer that is shared amongst all groups of this atom type.
    pub type_synchronizer: &'a Synchronizer<T>,
    /// A synchronizer that is shared amongst all leading groups in all images.
    pub estimators_barrier: &'a Barrier,
    /// An adder for classical estimator calculations.
    pub system_adder: SysAdd,
    /// An adder for quantum estimator calculations.
    pub image_adder: ImAdd,
    /// An adder for averaging-out quantum estimator calculations.
    pub estimators_adder: EstAdd,
    /// A multiplier for classical estimator calculations.
    pub system_multiplier: SysMul,
    /// A multiplier for quantum estimator calculations.
    pub image_multiplier: ImMul,
    /// The physical potential.
    pub physical_potential: Phys,
    /// The exchange potential.
    pub exchange_potential: Stat<Dist, Boson>,
    /// The thermostat.
    pub thermostat: Therm,
    /// The time propagator.
    pub propagator: Prop,
}

impl<'a, T, SysAdd, ImAdd, EstAdd, SysMul, ImMul, Phys, Dist, Boson, Therm, Prop>
    Simulation<'a, T, SysAdd, ImAdd, EstAdd, SysMul, ImMul, Phys, Dist, Boson, Therm, Prop>
{
    pub fn step_trailing_group<V, OutPhys, OutExch, EstErr, Err>(
        &mut self,
        step: usize,
        quantum_estimators: Option<
            &mut [Estimator<
                impl QuantumEstimator<T, V, ImAdd, ImMul, (), Output = T, Error = EstErr>,
                impl QuantumEstimator<T, V, ImAdd, ImMul, (), Output = V, Error = EstErr>,
            >],
        >,
        classical_estimators: Option<
            &mut [Estimator<
                impl ClassicalEstimator<T, V, SysAdd, SysMul, (), Output = T, Error = EstErr>,
                impl ClassicalEstimator<T, V, SysAdd, SysMul, (), Output = V, Error = EstErr>,
            >],
        >,
        positions: &mut GroupRwLockInTypeInImageInSystem<V>,
        momenta: &mut GroupRwLockInTypeInImageInSystem<V>,
        physical_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
        exchange_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
    ) -> Result<(), Err>
    where
        Phys: PhysicalPotential<T, V, OutPhys>,
        Dist: ExchangePotential<T, V, OutExch> + Distinguishable,
        Boson: ExchangePotential<T, V, OutExch> + Bosonic,
        Therm: Thermostat<T, V>,
        Prop: Propagator<T, V, Phys, Dist, Boson, Therm, OutPhys, OutExch>,
        T: Clone,
        OutPhys: ValidOutput<T> + PropagationOutput<T>,
        OutExch: ValidOutput<T> + PropagationOutput<T>,
        Err: From<CommError> + From<Prop::Error> + From<EstErr>,
    {
        let (physical_potential_energy, type_exchange_potential_energy, group_heat) =
            self.propagator.propagate(
                step,
                &mut self.physical_potential,
                self.exchange_potential.as_mut(),
                &mut self.thermostat,
                positions,
                momenta,
                physical_forces,
                exchange_forces,
            )?;

        if quantum_estimators.is_some() || classical_estimators.is_some() {
            let physical_potential_energy = physical_potential_energy
                .get(self.image_synchronizer, || {
                    CommError::new(self.image, self.group)
                })?;

            let type_exchange_potential_energy = type_exchange_potential_energy
                .get(self.type_synchronizer, || {
                    CommError::new(self.image, self.group)
                })?;

            if let Some(estimators) = quantum_estimators {
                for estimator in estimators {
                    match estimator {
                        Estimator::Value(estimator) => {
                            estimator.calculate(
                                &mut self.image_synchronizer,
                                &mut self.image_adder,
                                &mut self.image_multiplier,
                                physical_potential_energy.clone(),
                                type_exchange_potential_energy.clone(),
                                &positions
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &physical_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &exchange_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                            )?;
                        }
                        Estimator::Vector(estimator) => {
                            estimator.calculate(
                                &mut self.image_synchronizer,
                                &mut self.image_adder,
                                &mut self.image_multiplier,
                                physical_potential_energy.clone(),
                                type_exchange_potential_energy.clone(),
                                &positions
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &physical_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &exchange_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                            )?;
                        }
                    }
                    self.image_synchronizer.barrier.wait();
                }
            }

            if let Some(estimators) = classical_estimators {
                for estimator in estimators {
                    match estimator {
                        Estimator::Value(estimator) => {
                            estimator.calculate(
                                &mut self.system_synchronizer,
                                &mut self.image_synchronizer,
                                &mut self.system_adder,
                                &mut self.system_multiplier,
                                physical_potential_energy.clone(),
                                type_exchange_potential_energy.clone(),
                                group_heat.clone(),
                                &positions
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &momenta
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &physical_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &exchange_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                            )?;
                        }
                        Estimator::Vector(estimator) => {
                            estimator.calculate(
                                &mut self.system_synchronizer,
                                &mut self.image_synchronizer,
                                &mut self.system_adder,
                                &mut self.system_multiplier,
                                physical_potential_energy.clone(),
                                type_exchange_potential_energy.clone(),
                                group_heat.clone(),
                                &positions
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &momenta
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &physical_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &exchange_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                            )?;
                        }
                    }
                    self.system_synchronizer.barrier.wait();
                }
            }
        }
        Ok(())
    }

    pub fn step_leading_group<const N: usize, V, OutPhys, OutExch, EstErr, Err>(
        &mut self,
        step: usize,
        quantum_estimators: Option<
            &mut [Estimator<
                impl QuantumEstimator<T, V, ImAdd, ImMul, T, Output = T, Error = EstErr>,
                impl QuantumEstimator<T, V, ImAdd, ImMul, V, Output = V, Error = EstErr>,
            >],
        >,
        classical_estimators: Option<
            &mut [Estimator<
                impl ClassicalEstimator<T, V, SysAdd, SysMul, (), Output = T, Error = EstErr>,
                impl ClassicalEstimator<T, V, SysAdd, SysMul, (), Output = V, Error = EstErr>,
            >],
        >,
        positions: &mut GroupRwLockInTypeInImageInSystem<V>,
        momenta: &mut GroupRwLockInTypeInImageInSystem<V>,
        physical_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
        exchange_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
    ) -> Result<(), Err>
    where
        EstAdd: SyncAddSender<T>,
        Phys: PhysicalPotential<T, V, OutPhys>,
        Dist: ExchangePotential<T, V, OutExch> + Distinguishable,
        Boson: ExchangePotential<T, V, OutExch> + Bosonic,
        Therm: Thermostat<T, V>,
        Prop: Propagator<T, V, Phys, Dist, Boson, Therm, OutPhys, OutExch>,
        T: Clone + MeaningfulOutput,
        V: Vector<N, Element = T> + MeaningfulOutput,
        OutPhys: ValidOutput<T> + PropagationOutput<T>,
        OutExch: ValidOutput<T> + PropagationOutput<T>,
        Err: From<CommError> + From<EstAdd::Error> + From<Prop::Error> + From<EstErr>,
    {
        let (physical_potential_energy, type_exchange_potential_energy, group_heat) =
            self.propagator.propagate(
                step,
                &mut self.physical_potential,
                self.exchange_potential.as_mut(),
                &mut self.thermostat,
                positions,
                momenta,
                physical_forces,
                exchange_forces,
            )?;

        if quantum_estimators.is_some() || classical_estimators.is_some() {
            let physical_potential_energy = physical_potential_energy
                .get(self.image_synchronizer, || {
                    CommError::new(self.image, self.group)
                })?;

            let type_exchange_potential_energy = type_exchange_potential_energy
                .get(self.type_synchronizer, || {
                    CommError::new(self.image, self.group)
                })?;

            if let Some(estimators) = quantum_estimators {
                for estimator in estimators {
                    match estimator {
                        Estimator::Value(estimator) => {
                            self.estimators_adder.send(
                                estimator.calculate(
                                    &mut self.image_synchronizer,
                                    &mut self.image_adder,
                                    &mut self.image_multiplier,
                                    physical_potential_energy.clone(),
                                    type_exchange_potential_energy.clone(),
                                    &positions
                                        .as_map_mut()
                                        .map_map(|map| map.read())
                                        .map_whole(|whole| whole.into()),
                                    &physical_forces
                                        .as_map_mut()
                                        .map_map(|map| map.read())
                                        .map_whole(|whole| whole.into()),
                                    &exchange_forces
                                        .as_map_mut()
                                        .map_map(|map| map.read())
                                        .map_whole(|whole| whole.into()),
                                )?,
                            )?;
                            self.estimators_barrier.wait();
                        }
                        Estimator::Vector(estimator) => {
                            let vector = estimator.calculate(
                                &mut self.image_synchronizer,
                                &mut self.image_adder,
                                &mut self.image_multiplier,
                                physical_potential_energy.clone(),
                                type_exchange_potential_energy.clone(),
                                &positions
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &physical_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &exchange_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                            )?;
                            for element in vector.as_array() {
                                self.estimators_adder.send(element.clone())?;
                                self.estimators_barrier.wait();
                            }
                        }
                    }
                    self.image_synchronizer.barrier.wait();
                }
            }

            if let Some(estimators) = classical_estimators {
                for estimator in estimators {
                    match estimator {
                        Estimator::Value(estimator) => {
                            estimator.calculate(
                                &mut self.system_synchronizer,
                                &mut self.image_synchronizer,
                                &mut self.system_adder,
                                &mut self.system_multiplier,
                                physical_potential_energy.clone(),
                                type_exchange_potential_energy.clone(),
                                group_heat.clone(),
                                &positions
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &momenta
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &physical_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &exchange_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                            )?;
                        }
                        Estimator::Vector(estimator) => {
                            estimator.calculate(
                                &mut self.system_synchronizer,
                                &mut self.image_synchronizer,
                                &mut self.system_adder,
                                &mut self.system_multiplier,
                                physical_potential_energy.clone(),
                                type_exchange_potential_energy.clone(),
                                group_heat.clone(),
                                &positions
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &momenta
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &physical_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                                &exchange_forces
                                    .as_map_mut()
                                    .map_map(|map| map.read())
                                    .map_whole(|whole| whole.into()),
                            )?;
                        }
                    }
                    self.system_synchronizer.barrier.wait();
                }
            }
        }
        Ok(())
    }

    pub fn step_output<const N: usize, V, ValS, VecS, OutPhys, OutExch, EstErr, Err>(
        &mut self,
        step: usize,
        estimators_output: EstimatorsOutputOption<
            &mut [Estimator<
                impl QuantumEstimator<T, V, ImAdd, ImMul, T, Output = T, Error = EstErr>,
                impl QuantumEstimator<T, V, ImAdd, ImMul, V, Output = V, Error = EstErr>,
            >],
            &mut [Estimator<
                impl ClassicalEstimator<T, V, SysAdd, SysMul, T, Output = T, Error = EstErr>,
                impl ClassicalEstimator<T, V, SysAdd, SysMul, V, Output = V, Error = EstErr>,
            >],
            &mut ValS,
        >,
        positions_output: Option<&mut VecS>,
        momenta_output: Option<&mut VecS>,
        physical_forces_output: Option<&mut VecS>,
        exchange_forces_output: Option<&mut VecS>,
        positions: &mut GroupRwLockInTypeInImageInSystem<V>,
        momenta: &mut GroupRwLockInTypeInImageInSystem<V>,
        physical_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
        exchange_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
    ) -> Result<(), Err>
    where
        EstAdd: SyncAddReceiver<T>,
        Phys: PhysicalPotential<T, V, OutPhys>,
        Dist: ExchangePotential<T, V, OutExch> + Distinguishable,
        Boson: ExchangePotential<T, V, OutExch> + Bosonic,
        Therm: Thermostat<T, V>,
        Prop: Propagator<T, V, Phys, Dist, Boson, Therm, OutPhys, OutExch>,
        T: Clone + Add<Output = T> + Div<Output = T> + From<usize> + MeaningfulOutput,
        V: Vector<N, Element = T> + MeaningfulOutput,
        ValS: ValuesStream<T> + ValuesStream<V>,
        VecS: VectorsStream<N, T, V>,
        OutPhys: ValidOutput<T> + PropagationOutput<T>,
        OutExch: ValidOutput<T> + PropagationOutput<T>,
        Err: From<CommError>
            + From<EstAdd::Error>
            + From<Prop::Error>
            + From<ValS::Error>
            + From<VecS::Error>
            + From<EstErr>,
    {
        let (physical_potential_energy, type_exchange_potential_energy, group_heat) =
            self.propagator.propagate(
                step,
                &mut self.physical_potential,
                self.exchange_potential.as_mut(),
                &mut self.thermostat,
                positions,
                momenta,
                physical_forces,
                exchange_forces,
            )?;

        match estimators_output {
            EstimatorsOutputOption::None => {}
            estimators_output @ _ => {
                let physical_potential_energy = physical_potential_energy
                    .get(self.image_synchronizer, || {
                        CommError::new(self.image, self.group)
                    })?;

                let type_exchange_potential_energy = type_exchange_potential_energy
                    .get(self.type_synchronizer, || {
                        CommError::new(self.image, self.group)
                    })?;

                macro_rules! write_observables {
                    (@quantum $estimators:expr, $stream:expr) => {
                        for estimator in $estimators {
                            match estimator {
                                Estimator::Value(estimator) => {
                                    let value = estimator.calculate(
                                        &self.image_synchronizer,
                                        &mut self.image_adder,
                                        &mut self.image_multiplier,
                                        physical_potential_energy.clone(),
                                        type_exchange_potential_energy.clone(),
                                        &positions
                                            .as_map_mut()
                                            .map_map(|map| map.read())
                                            .map_whole(|whole| whole.into()),
                                        &physical_forces
                                            .as_map_mut()
                                            .map_map(|map| map.read())
                                            .map_whole(|whole| whole.into()),
                                        &exchange_forces
                                            .as_map_mut()
                                            .map_map(|map| map.read())
                                            .map_whole(|whole| whole.into()),
                                    )?;
                                    self.estimators_barrier.wait();
                                    $stream.write_value(
                                        match self.estimators_adder.recv_sum()? {
                                            Some(other) => value + other,
                                            None => value,
                                        } / self.images.into(),
                                    )?;
                                }
                                Estimator::Vector(estimator) => {
                                    let mut vector = estimator.calculate(
                                        &self.image_synchronizer,
                                        &mut self.image_adder,
                                        &mut self.image_multiplier,
                                        physical_potential_energy.clone(),
                                        type_exchange_potential_energy.clone(),
                                        &positions
                                            .as_map_mut()
                                            .map_map(|map| map.read())
                                            .map_whole(|whole| whole.into()),
                                        &physical_forces
                                            .as_map_mut()
                                            .map_map(|map| map.read())
                                            .map_whole(|whole| whole.into()),
                                        &exchange_forces
                                            .as_map_mut()
                                            .map_map(|map| map.read())
                                            .map_whole(|whole| whole.into()),
                                    )?;
                                    for element in vector.as_mut_array() {
                                        self.estimators_barrier.wait();
                                        if let Some(other) = self.estimators_adder.recv_sum()? {
                                            *element =
                                                (element.clone() + other) / self.images.into();
                                        }
                                    }
                                    $stream.write_value(vector)?;
                                }
                            }
                            self.system_synchronizer.barrier.wait();
                        }
                    };
                    (@classical $estimators:expr, $stream:expr) => {
                        for estimator in $estimators {
                            match estimator {
                                Estimator::Value(estimator) => {
                                    $stream.write_value(
                                        estimator.calculate(
                                            &self.system_synchronizer,
                                            &self.image_synchronizer,
                                            &mut self.system_adder,
                                            &mut self.system_multiplier,
                                            physical_potential_energy.clone(),
                                            type_exchange_potential_energy.clone(),
                                            group_heat.clone(),
                                            &positions
                                                .as_map_mut()
                                                .map_map(|map| map.read())
                                                .map_whole(|whole| whole.into()),
                                            &momenta
                                                .as_map_mut()
                                                .map_map(|map| map.read())
                                                .map_whole(|whole| whole.into()),
                                            &physical_forces
                                                .as_map_mut()
                                                .map_map(|map| map.read())
                                                .map_whole(|whole| whole.into()),
                                            &exchange_forces
                                                .as_map_mut()
                                                .map_map(|map| map.read())
                                                .map_whole(|whole| whole.into()),
                                        )?,
                                    )?;
                                }
                                Estimator::Vector(estimator) => {
                                    $stream.write_value(
                                        estimator.calculate(
                                            &self.system_synchronizer,
                                            &self.image_synchronizer,
                                            &mut self.system_adder,
                                            &mut self.system_multiplier,
                                            physical_potential_energy.clone(),
                                            type_exchange_potential_energy.clone(),
                                            group_heat.clone(),
                                            &positions
                                                .as_map_mut()
                                                .map_map(|map| map.read())
                                                .map_whole(|whole| whole.into()),
                                            &momenta
                                                .as_map_mut()
                                                .map_map(|map| map.read())
                                                .map_whole(|whole| whole.into()),
                                            &physical_forces
                                                .as_map_mut()
                                                .map_map(|map| map.read())
                                                .map_whole(|whole| whole.into()),
                                            &exchange_forces
                                                .as_map_mut()
                                                .map_map(|map| map.read())
                                                .map_whole(|whole| whole.into()),
                                        )?,
                                    )?;
                                }
                            }
                            self.system_synchronizer.barrier.wait();
                        }
                    };
                }

                match estimators_output {
                    EstimatorsOutputOption::Quantum(EstimatorsOutput { estimators, stream }) => {
                        stream.write_prelude(step)?;
                        write_observables!(@quantum estimators, stream);
                        stream.new_line()?;
                    }
                    EstimatorsOutputOption::Classical(EstimatorsOutput { estimators, stream }) => {
                        stream.write_prelude(step)?;
                        write_observables!(@classical estimators, stream);
                        stream.new_line()?;
                    }
                    EstimatorsOutputOption::Shared {
                        quantum_estimators,
                        classical_estimators,
                        stream,
                    } => {
                        stream.write_prelude(step)?;
                        write_observables!(@quantum quantum_estimators, stream);
                        write_observables!(@classical classical_estimators, stream);
                        stream.new_line()?;
                    }
                    EstimatorsOutputOption::Separate { quantum, classical } => {
                        quantum.stream.write_prelude(step)?;
                        write_observables!(@quantum quantum.estimators, quantum.stream);
                        quantum.stream.new_line()?;

                        classical.stream.write_prelude(step)?;
                        write_observables!(@classical classical.estimators, classical.stream);
                        classical.stream.new_line()?;
                    }
                    EstimatorsOutputOption::None => unreachable!(),
                }
            }
        }

        macro_rules! write_vectors {
            ($stream:expr, $vectors:expr) => {
                $stream.write_prelude(step)?;
                $stream.write_vectors(
                    &$vectors
                        .as_map_mut()
                        .map_map(|map| map.read())
                        .map_whole(|whole| whole.into()),
                )?;
                $stream.new_line()?;
            };
        }

        if let Some(stream) = positions_output {
            write_vectors!(stream, positions);
        }

        if let Some(stream) = momenta_output {
            write_vectors!(stream, momenta);
        }

        if let Some(stream) = physical_forces_output {
            write_vectors!(stream, physical_forces);
        }

        if let Some(stream) = exchange_forces_output {
            write_vectors!(stream, exchange_forces);
        }

        Ok(())
    }
}
