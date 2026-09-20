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
        core::stat::{Bosonic, Distinguishable},
        thermostat::Thermostat,
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
}

pub use unimplemented::{Unimplemented, UnimplementedError};
