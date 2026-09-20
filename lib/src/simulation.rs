use crate::{
    core::{
        GroupImageInfo, GroupRwLockInTypeInImageInSystem, Synchronizer, Vector,
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
    sync::RwLock,
};

pub trait PropagationOutput<T> {
    fn get<F, E>(self, synchronizer: &Synchronizer<RwLock<T>>, f: F) -> Result<T, E>
    where
        F: FnOnce() -> E;
}

impl<T: Clone> PropagationOutput<T> for () {
    fn get<F, E>(self, synchronizer: &Synchronizer<RwLock<T>>, f: F) -> Result<T, E>
    where
        F: FnOnce() -> E,
    {
        synchronizer.barrier.wait();
        synchronizer
            .sync
            .read()
            .map(|guard| guard.clone())
            .map_err(|_| f())
    }
}

impl<T: Clone + MeaningfulOutput> PropagationOutput<T> for T {
    fn get<F, E>(self, synchronizer: &Synchronizer<RwLock<T>>, f: F) -> Result<T, E>
    where
        F: FnOnce() -> E,
    {
        synchronizer
            .sync
            .write()
            .map(|mut guard| *guard = self.clone())
            .map_err(|_| f())?;
        synchronizer.barrier.wait();
        Ok(self)
    }
}

pub struct SimTrailing<T, SysAdd, ImAdd, EstAdd, SysMul, ImMul, Phys, Dist, Boson, Therm, Prop> {
    images: usize,
    group_and_image: GroupImageInfo,
    system_synchronizer: Synchronizer<RwLock<T>>,
    image_synchronizer: Synchronizer<RwLock<T>>,
    type_synchronizer: Synchronizer<RwLock<T>>,
    system_adder: SysAdd,
    image_adder: ImAdd,
    estimators_adder: Synchronizer<EstAdd>,
    system_multiplier: SysMul,
    image_multiplier: ImMul,
    physical_potential: Phys,
    exchange_potential: Stat<Dist, Boson>,
    thermostat: Therm,
    propagator: Prop,
}

impl<T, SysAdd, ImAdd, EstAdd, SysMul, ImMul, Phys, Dist, Boson, Therm, Prop>
    SimTrailing<T, SysAdd, ImAdd, EstAdd, SysMul, ImMul, Phys, Dist, Boson, Therm, Prop>
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
                .get(&self.image_synchronizer, || {
                    CommError::from(self.group_and_image)
                })?;

            let type_exchange_potential_energy = type_exchange_potential_energy
                .get(&self.type_synchronizer, || {
                    CommError::from(self.group_and_image)
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
        Err: From<CommError> + From<Prop::Error> + From<EstErr> + From<EstAdd::Error>,
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
                .get(&self.image_synchronizer, || {
                    CommError::from(self.group_and_image)
                })?;

            let type_exchange_potential_energy = type_exchange_potential_energy
                .get(&self.type_synchronizer, || {
                    CommError::from(self.group_and_image)
                })?;

            if let Some(estimators) = quantum_estimators {
                for estimator in estimators {
                    match estimator {
                        Estimator::Value(estimator) => {
                            self.estimators_adder.sync.send(
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
                            self.estimators_adder.barrier.wait();
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
                                self.estimators_adder.sync.send(element.clone())?;
                                self.estimators_adder.barrier.wait();
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
            + From<Prop::Error>
            + From<EstErr>
            + From<ValS::Error>
            + From<EstAdd::Error>
            + From<VecS::Error>,
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
                    .get(&self.image_synchronizer, || {
                        CommError::from(self.group_and_image)
                    })?;

                let type_exchange_potential_energy = type_exchange_potential_energy
                    .get(&self.type_synchronizer, || {
                        CommError::from(self.group_and_image)
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
                                    self.estimators_adder.barrier.wait();
                                    $stream.write_value(
                                        match self.estimators_adder.sync.recv_sum()? {
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
                                        self.estimators_adder.barrier.wait();
                                        if let Some(other) =
                                            self.estimators_adder.sync.recv_sum()?
                                        {
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
                );
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
