mod virial_kinetic_energy {
    use std::{
        convert::Infallible,
        error::Error,
        ops::{Add, Mul},
    };

    pub struct VirialKineticEnergy<const N: usize>;

    impl<const N: usize> VirialKineticEnergy<N> {
        pub fn new() -> Self {
            Self
        }
    }
}

pub use virial_kinetic_energy::VirialKineticEnergy;

mod primitive_kinetic_energy {
    pub struct PrimitiveKineticEnergy<const N: usize>;
}
