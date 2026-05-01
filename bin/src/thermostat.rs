mod langevin {
    use std::{array, convert::Infallible, ops::Mul};

    use lib::{
        core::{Decoupled, Vector, error::EmptyError},
        thermostat::AtomDecoupledThermostat,
    };
    use num::Float;
    use rand::Rng;
    use rand_distr::{Distribution, StandardNormal};

    use crate::core::constants::BOLTZMANN_CONSTANT;

    pub struct Langevin<const N: usize, T, R> {
        mass: T,
        beta_recip: T,
        gamma: T,
        rng: R,
    }

    impl<const N: usize, T, R> Langevin<N, T, R>
    where
        T: Clone + From<f32> + PartialOrd + Mul<Output = T>,
    {
        pub fn new(mass: T, temperature: T, gamma: T, rng: R) -> Decoupled<Self> {
            assert!(mass.clone() > 0.0.into(), "the mass must be positive");
            assert!(
                temperature.clone() > 0.0.into(),
                "the temperature must be positive"
            );
            Decoupled::new(Self {
                mass,
                beta_recip: T::from(BOLTZMANN_CONSTANT) * temperature,
                gamma,
                rng,
            })
        }
    }

    impl<const N: usize, T, V, R> AtomDecoupledThermostat<T, V> for Langevin<N, T, R>
    where
        T: Clone + From<f32> + Float,
        V: Vector<N, Element = T> + Clone,
        R: Rng,
    {
        type ErrorAtom = Infallible;
        type ErrorSystem = EmptyError;

        fn thermalize(
            &mut self,
            step_size: T,
            _atom_index: usize,
            _position: &V,
            _physical_force: &V,
            _exchange_force: &V,
            momentum: &mut V,
        ) -> Result<T, Self::ErrorAtom> {
            let gamma_times_dt = self.gamma.clone() * step_size;
            let momentum_old = momentum.clone();
            let momentum_new = momentum_old.clone()
                * (<T as From<_>>::from(-0.5) * gamma_times_dt.clone()).exp()
                + V::from(array::from_fn(|_| {
                    <T as From<_>>::from(StandardNormal.sample(&mut self.rng))
                })) * (self.mass.clone() * self.beta_recip.clone() * -(-gamma_times_dt).exp_m1())
                    .sqrt();
            *momentum = momentum_new.clone();
            Ok(<T as From<_>>::from(0.5) / self.mass.clone()
                * (momentum_new.magnitude_squared() - momentum_old.magnitude_squared()))
        }
    }
}

pub use langevin::Langevin;
