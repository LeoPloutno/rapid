pub mod constants {
    pub const REDUCED_PLANK_CONSTANT: f32 = 1.0;
    pub const BOLTZMANN_CONSTANT: f32 = 1.0;
}

mod unimplemented {
    use std::{
        error::Error,
        fmt::{Display, Formatter, Result as FmtResult},
    };

    use arc_rw_lock::ElementRwLock;
    use lib::{
        ImageHandle,
        core::{
            marker::{InnerIsLeading, InnerIsTrailing},
            stat::{Bosonic, Distinguishable},
        },
        potential::{
            exchange::{
                InnerExchangePotential,
                quadratic::{InnerNormalModesTransform, InnerQuadraticExpansionExchangePotential},
            },
            physical::{AtomDecoupledPhysicalPotential, PhysicalPotential},
        },
        propagator::{InnerPropagator, quadratic::InnerQuadraticExpansionPropagator},
        thermostat::{AtomDecoupledThermostat, Thermostat},
    };

    #[derive(Clone, Copy, Debug)]
    pub struct Unimplemented;

    impl InnerIsLeading for Unimplemented {}

    impl InnerIsTrailing for Unimplemented {}

    impl Distinguishable for Unimplemented {}

    impl Bosonic for Unimplemented {}

    impl<T, V> AtomDecoupledPhysicalPotential<T, V> for Unimplemented {
        type Error = UnimplementedError;

        fn calculate_potential_set_force(
            &mut self,
            _atom_index: usize,
            _position: &V,
            _force: &mut V,
        ) -> Result<T, Self::Error> {
            Err(UnimplementedError)
        }

        fn calculate_potential_add_force(
            &mut self,
            _atom_index: usize,
            _position: &V,
            _force: &mut V,
        ) -> Result<T, Self::Error> {
            Err(UnimplementedError)
        }

        fn calculate_potential(&mut self, _atom_index: usize, _position: &V) -> Result<T, Self::Error> {
            Err(UnimplementedError)
        }

        fn set_force(&mut self, _atom_index: usize, _position: &V, _force: &mut V) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }

        fn add_force(&mut self, _atom_index: usize, _position: &V, _force: &mut V) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }
    }

    impl<T, V> InnerExchangePotential<T, V> for Unimplemented {
        type Error = UnimplementedError;

        fn calculate_potential_set_forces(
            &mut self,
            _type_positions_prev_image: &[V],
            _type_positions_next_image: &[V],
            _type_positions: &[V],
            _group_forces: &mut [V],
        ) -> Result<T, Self::Error> {
            Err(UnimplementedError)
        }

        fn calculate_potential_add_forces(
            &mut self,
            _type_positions_prev_image: &[V],
            _type_positions_next_image: &[V],
            _type_positions: &[V],
            _group_forces: &mut [V],
        ) -> Result<T, Self::Error> {
            Err(UnimplementedError)
        }

        fn calculate_potential(
            &mut self,
            _type_positions_prev_image: &[V],
            _type_positions_next_image: &[V],
            _type_positions: &[V],
        ) -> Result<T, Self::Error> {
            Err(UnimplementedError)
        }

        fn set_forces(
            &mut self,
            _type_positions_prev_image: &[V],
            _type_positions_next_image: &[V],
            _type_positions: &[V],
            _group_forces: &mut [V],
        ) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }

        fn add_forces(
            &mut self,
            _type_positions_prev_image: &[V],
            _type_positions_next_image: &[V],
            _type_positions: &[V],
            _group_forces: &mut [V],
        ) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }
    }

    impl<'a, T, V> InnerQuadraticExpansionExchangePotential<'a, T, V> for Unimplemented {
        type QuadraticPotential = Self;
        type ResiduePotential = Self;

        fn as_quadratic_expansion(&'a mut self) -> (Self::QuadraticPotential, Self::ResiduePotential) {
            (Self, Self)
        }
    }

    impl<T, V> InnerNormalModesTransform<T, V> for Unimplemented {
        type Error = UnimplementedError;

        fn cartesian_to_normal_modes(
            &mut self,
            _images_type_positions: &ElementRwLock<lib::core::GroupImageHandle<V>>,
            _normal_modes: &mut [V],
        ) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }

        fn normal_modes_to_cartesian(
            &mut self,
            _images_normal_modes: &ElementRwLock<arc_rw_lock::UniqueArcSliceRwLock<V>>,
            _group_position: &mut [V],
        ) -> Result<(), Self::Error> {
            Err(UnimplementedError)
        }

        fn eigenvalues(&self, _eigenvalues: &mut [T]) {}
    }

    impl<T, V> AtomDecoupledThermostat<T, V> for Unimplemented {
        type Error = UnimplementedError;

        fn thermalize(
            &mut self,
            _step_size: T,
            _atom_index: usize,
            _position: &V,
            _physical_force: &V,
            _exchange_force: &V,
            _momentum: &mut V,
        ) -> Result<T, Self::Error> {
            Err(UnimplementedError)
        }
    }

    impl<T, V, Phys, Dist, Boson, Therm> InnerPropagator<T, V, Phys, Dist, Boson, Therm> for Unimplemented
    where
        Phys: PhysicalPotential<T, V> + ?Sized,
        Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
        Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
        Therm: Thermostat<T, V> + ?Sized,
    {
        type Error = UnimplementedError;

        fn propagate(
            &mut self,
            _step: usize,
            _physical_potential: &mut Phys,
            _exchange_potential: lib::core::stat::Stat<&mut Dist, &mut Boson>,
            _thermostat: &mut Therm,
            _groups_positions: &mut ImageHandle<V>,
            _groups_momenta: &mut ImageHandle<V>,
            _groups_physical_forces: &mut ImageHandle<V>,
            _groups_exchange_forces: &mut ImageHandle<V>,
        ) -> Result<(T, T, T), Self::Error> {
            Err(UnimplementedError)
        }
    }

    impl<T, V, Phys, Dist, Boson, Therm> InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> for Unimplemented
    where
        Phys: PhysicalPotential<T, V> + ?Sized,
        Dist: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
        Boson: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
        Therm: Thermostat<T, V> + ?Sized,
    {
        type Error = UnimplementedError;

        fn propagate(
            &mut self,
            _step: usize,
            _physical_potential: &mut Phys,
            _exchange_potential: lib::core::stat::Stat<&mut Dist, &mut Boson>,
            _thermostat: &mut Therm,
            _groups_positions: &mut ImageHandle<V>,
            _groups_momenta: &mut ImageHandle<V>,
            _groups_physical_forces: &mut ImageHandle<V>,
            _groups_exchange_forces: &mut ImageHandle<V>,
        ) -> Result<(T, T, T), Self::Error> {
            Err(UnimplementedError)
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct UnimplementedError;

    impl Display for UnimplementedError {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            write!(f, "not implemented")
        }
    }

    impl Error for UnimplementedError {}
}

pub mod error {
    use std::{
        error::Error,
        fmt::{Display, Formatter, Result as FmtResult},
    };

    #[derive(Clone, Copy, Debug)]
    pub struct NumCastError;

    impl Display for NumCastError {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            write!(f, "failed to cast from a primitive")
        }
    }

    impl Error for NumCastError {}
}

pub use unimplemented::{Unimplemented, UnimplementedError};
