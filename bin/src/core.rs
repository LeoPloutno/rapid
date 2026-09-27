pub mod constants {
    pub const REDUCED_PLANK_CONSTANT: f32 = 1.0;
    pub const BOLTZMANN_CONSTANT: f32 = 1.0;
}

mod unimplemented {
    use lib::{
        core::{
            GroupInTypeInImage, GroupInTypeInImageInSystem,
            marker::ValidOutput,
            stat::{Bosonic, Distinguishable},
        },
        potential::{exchange::ExchangePotential, physical::PhysicalPotential},
        thermostat::Thermostat,
    };
    use std::{
        error::Error,
        fmt::{Display, Formatter, Result as FmtResult},
    };

    #[derive(Clone, Copy, Debug)]
    pub struct Unimplemented;

    impl Distinguishable for Unimplemented {}

    impl Bosonic for Unimplemented {}

    #[derive(Clone, Copy, Debug)]
    pub struct UnimplementedError;

    impl Display for UnimplementedError {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            write!(f, "not implemented")
        }
    }

    impl Error for UnimplementedError {}

    impl<T, V, O: ValidOutput<T>> PhysicalPotential<T, V, O> for Unimplemented {
        type Error = UnimplementedError;

        #[inline]
        fn calculate_energy_set_forces(
            &mut self,
            _positions: &GroupInTypeInImage<V>,
            _forces: &mut [V],
        ) -> Result<O, Self::Error> {
            Err(UnimplementedError)
        }

        #[inline]
        fn calculate_energy_add_forces(
            &mut self,
            _positions: &GroupInTypeInImage<V>,
            _forces: &mut [V],
        ) -> Result<O, Self::Error> {
            Err(UnimplementedError)
        }

        #[inline]
        fn calculate_energy(
            &mut self,
            _positions: &GroupInTypeInImage<V>,
        ) -> Result<O, Self::Error> {
            Err(UnimplementedError)
        }

        #[inline]
        fn set_forces(
            &mut self,
            _positions: &GroupInTypeInImage<V>,
            _forces: &mut [V],
        ) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }

        #[inline]
        fn add_forces(
            &mut self,
            _positions: &GroupInTypeInImage<V>,
            _forces: &mut [V],
        ) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }
    }

    impl<T, V, O: ValidOutput<T>> ExchangePotential<T, V, O> for Unimplemented {
        type Error = UnimplementedError;

        fn calculate_energy_set_forces(
            &mut self,
            _prev_image_positions: &GroupInTypeInImage<V>,
            _next_image_positions: &GroupInTypeInImage<V>,
            _positions: &GroupInTypeInImage<V>,
            _forces: &mut [V],
        ) -> Result<O, Self::Error> {
            Err(UnimplementedError)
        }

        fn calculate_energy_add_forces(
            &mut self,
            _prev_image_positions: &GroupInTypeInImage<V>,
            _next_image_positions: &GroupInTypeInImage<V>,
            _positions: &GroupInTypeInImage<V>,
            _forces: &mut [V],
        ) -> Result<O, Self::Error> {
            Err(UnimplementedError)
        }

        fn calculate_energy(
            &mut self,
            _prev_image_positions: &GroupInTypeInImage<V>,
            _next_image_positions: &GroupInTypeInImage<V>,
            _positions: &GroupInTypeInImage<V>,
        ) -> Result<O, Self::Error> {
            Err(UnimplementedError)
        }

        fn set_forces(
            &mut self,
            _prev_image_positions: &GroupInTypeInImage<V>,
            _next_image_positions: &GroupInTypeInImage<V>,
            _positions: &GroupInTypeInImage<V>,
            _forces: &mut [V],
        ) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }

        fn add_forces(
            &mut self,
            _prev_image_positions: &GroupInTypeInImage<V>,
            _next_image_positions: &GroupInTypeInImage<V>,
            _positions: &GroupInTypeInImage<V>,
            _forces: &mut [V],
        ) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }
    }

    impl<T, V> Thermostat<T, V> for Unimplemented {
        type Error = UnimplementedError;

        fn thermalize(
            &mut self,
            _positions: &GroupInTypeInImageInSystem<V>,
            _physical_forces: &GroupInTypeInImageInSystem<V>,
            _exchange_forces: &GroupInTypeInImageInSystem<V>,
            _momenta: &mut [V],
        ) -> Result<T, Self::Error> {
            Err(UnimplementedError)
        }
    }
}

pub use unimplemented::{Unimplemented, UnimplementedError};
