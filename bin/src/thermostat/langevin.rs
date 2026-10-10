use crate::core::constants::BOLTZMANN_CONSTANT;
use lib::{
    core::{Vector, error::EmptyError},
    thermostat::{AtomDecoupledThermostat, DecoupledThermostat},
};
use num::Float;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::{array, convert::Infallible};

pub struct Langevin<T, R> {
    mass: T,
    beta_recip: T,
    gamma: T,
    rng: R,
}

impl<T, R> Langevin<T, R>
where
    T: From<f32> + Float,
{
    pub fn new(mass: T, temperature: T, gamma: T, rng: R) -> DecoupledThermostat<Self> {
        assert!(mass > T::zero(), "the mass must be positive");
        assert!(temperature > T::zero(), "the temperature must be positive");
        assert!(gamma > T::zero(), "`gamma` must be positive");

        DecoupledThermostat::new(Self {
            mass: mass,
            beta_recip: <T as From<_>>::from(BOLTZMANN_CONSTANT) * temperature,
            gamma,
            rng,
        })
    }
}

impl<T, V, R> AtomDecoupledThermostat<T, V> for Langevin<T, R>
where
    T: From<f32> + Float,
    V: From<[T; V::DIM]> + Vector<Element = T>,
    R: Rng,
{
    type ErrorAtom = Infallible;
    type ErrorSystem = EmptyError;

    fn thermalize(
        &mut self,
        step_size: T,
        _step: usize,
        _atom_index: usize,
        _position: &V,
        _physical_force: &V,
        _exchange_force: &V,
        momentum: &mut V,
    ) -> Result<T, Self::ErrorAtom> {
        let gamma_times_dt = self.gamma * step_size;
        let momentum_old = *momentum;
        let momentum_new = momentum_old * (<T as From<_>>::from(-0.5) * gamma_times_dt).exp()
            + V::from(array::from_fn(|_| {
                <T as From<_>>::from(StandardNormal.sample(&mut self.rng))
            })) * (self.mass * self.beta_recip * -(-gamma_times_dt).exp_m1()).sqrt();
        *momentum = momentum_new;
        Ok(<T as From<_>>::from(0.5) / self.mass
            * (momentum_new.magnitude_squared() - momentum_old.magnitude_squared()))
    }
}
