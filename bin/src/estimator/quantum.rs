mod virial_kinetic_energy {
    use std::{
        convert::Infallible,
        error::Error,
        ops::{Add, Mul},
    };

    use lib::{
        core::{
            Vector,
            marker::{InnerIsLeading, InnerIsTrailing},
            stat::{Bosonic, Distinguishable},
            sync_ops::{SyncAddReciever, SyncAddSender},
        },
        estimator::quantum::atom_additive::{InnerAtomAdditiveQuantumEstimator, MainAtomAdditiveQuantumEstimator},
        potential::exchange::{InnerExchangePotential, quadratic::InnerQuadraticExpansionExchangePotential},
    };

    pub struct VirialKineticEnergy<const N: usize>;

    impl<const N: usize> VirialKineticEnergy<N> {
        pub fn new() -> Self {
            Self
        }
    }

    impl<const N: usize> InnerIsLeading for VirialKineticEnergy<N> {}

    impl<const N: usize> InnerIsTrailing for VirialKineticEnergy<N> {}

    impl<const N: usize, T, V, Adder> MainAtomAdditiveQuantumEstimator<T, V, Adder> for VirialKineticEnergy<N>
    where
        Adder: SyncAddReciever<T, Error: Error + 'static> + ?Sized,
    {
        type Output = T;
        type Error = Box<dyn Error + 'static>;
    }

    impl<const N: usize, T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
        InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for VirialKineticEnergy<N>
    where
        T: Clone + From<f32> + Add<Output = T> + Mul<Output = T>,
        V: Vector<N, Element = T> + Clone,
        Adder: SyncAddSender<T, Error: Error + 'static> + ?Sized,
        Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
        DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
        Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
        BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    {
        type Output = T;
        type ErrorAtom = Infallible;
        type ErrorSystem = Box<dyn Error + 'static>;

        fn calculate(
            &mut self,
            _atom_index: usize,
            _exchange_potential: lib::core::Scheme<
                lib::core::stat::Stat<&Dist, &Boson>,
                lib::core::stat::Stat<&DistQuad, &BosonQuad>,
            >,
            _group_physical_potential_energy: T,
            _group_exchange_potential_energy: T,
            position: &V,
            physical_force: &V,
            _exchange_force: &V,
        ) -> Result<Self::Output, Self::ErrorAtom> {
            Ok(T::from(-0.5) * position.clone().dot(physical_force.clone()))
        }
    }
}

pub use virial_kinetic_energy::VirialKineticEnergy;

mod primitive_kinetic_energy {
    pub struct PrimitiveKineticEnergy<const N: usize>;
}
