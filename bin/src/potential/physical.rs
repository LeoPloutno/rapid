mod harmonic {
    use std::{
        convert::Infallible,
        ops::{Add, Div, Mul},
    };

    use lib::{
        core::{Additive, Vector, error::EmptyError},
        potential::physical::AtomAdditivePhysicalPotential,
    };

    pub struct Harmonic<const N: usize, T> {
        potential_prefactor: T,
    }

    impl<const N: usize, T> Harmonic<N, T>
    where
        T: Clone + From<f32> + PartialOrd + Div<Output = T>,
    {
        pub fn new(spring_constant: T, inner_images: usize) -> Additive<Self> {
            assert!(
                spring_constant.clone() >= 0.0.into(),
                "spring constant must be non-negative"
            );
            Additive::new(Self {
                potential_prefactor: spring_constant / ((inner_images + 2) as f32).into(),
            })
        }
    }

    impl<const N: usize, T, V> AtomAdditivePhysicalPotential<T, V> for Harmonic<N, T>
    where
        T: Clone + From<f32> + Add<Output = T> + Mul<Output = T>,
        V: Vector<N, Element = T> + Clone,
    {
        type ErrorAtom = Infallible;
        type ErrorSystem = EmptyError;

        fn calculate_potential_set_force(
            &mut self,
            atom_index: usize,
            position: &V,
            force: &mut V,
        ) -> Result<T, Self::ErrorAtom> {
            #![allow(deprecated)]
            self.set_force(atom_index, position, force)?;
            Ok(self.calculate_potential(atom_index, position)?)
        }

        fn calculate_potential_add_force(
            &mut self,
            atom_index: usize,
            position: &V,
            force: &mut V,
        ) -> Result<T, Self::ErrorAtom> {
            #![allow(deprecated)]
            self.add_force(atom_index, position, force)?;
            Ok(self.calculate_potential(atom_index, position)?)
        }

        fn calculate_potential(
            &mut self,
            _atom_index: usize,
            position: &V,
        ) -> Result<T, Self::ErrorAtom> {
            Ok(self.potential_prefactor.clone() * position.clone().magnitude_squared())
        }

        fn set_force(
            &mut self,
            _atom_index: usize,
            position: &V,
            force: &mut V,
        ) -> Result<(), Self::ErrorAtom> {
            *force = -position.clone() * 2.0.into() * self.potential_prefactor.clone();
            Ok(())
        }

        fn add_force(
            &mut self,
            _atom_index: usize,
            position: &V,
            force: &mut V,
        ) -> Result<(), Self::ErrorAtom> {
            *force += -position.clone() * 2.0.into() * self.potential_prefactor.clone();
            Ok(())
        }
    }
}

pub use harmonic::Harmonic;
