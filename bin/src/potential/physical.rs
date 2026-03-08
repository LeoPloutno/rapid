mod harmonic {
    use std::{convert::Infallible, ops::Mul};

    use lib::{core::Vector, potential::physical::AtomDecoupledPhysicalPotential};

    pub struct Harmonic<const N: usize, T> {
        pub spring_constant: T,
    }

    impl<const N: usize, T, V> AtomDecoupledPhysicalPotential<T, V> for Harmonic<N, T>
    where
        T: From<f32> + Mul<Output = T> + Clone,
        V: Vector<N, Element = T> + Clone,
    {
        type Error = Infallible;

        fn calculate_potential_set_force(
            &mut self,
            atom_index: usize,
            position: &V,
            force: &mut V,
        ) -> Result<T, Self::Error> {
            #![allow(deprecated)]
            self.set_force(atom_index, position, force)?;
            self.calculate_potential(atom_index, position)
        }

        fn calculate_potential_add_force(
            &mut self,
            atom_index: usize,
            position: &V,
            force: &mut V,
        ) -> Result<T, Self::Error> {
            #![allow(deprecated)]
            self.add_force(atom_index, position, force)?;
            self.calculate_potential(atom_index, position)
        }

        fn calculate_potential(&mut self, _atom_index: usize, position: &V) -> Result<T, Self::Error> {
            Ok(T::from(0.5) * self.spring_constant.clone() * position.clone().magnitude_squared())
        }

        fn set_force(&mut self, _atom_index: usize, position: &V, force: &mut V) -> Result<(), Self::Error> {
            *force = -position.clone() * self.spring_constant.clone();
            Ok(())
        }

        fn add_force(&mut self, _atom_index: usize, position: &V, force: &mut V) -> Result<(), Self::Error> {
            *force += -position.clone() * self.spring_constant.clone();
            Ok(())
        }
    }
}
