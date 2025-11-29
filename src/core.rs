/// Information about atoms of the same type.
pub struct AtomGroupInfo<T> {
    /// Unique identifier.
    id: usize,
    /// The mass of a single atom in this group.
    mass: T,
}

#[cfg(feature = "monte_carlo")]
#[derive(Clone, Copy)]
pub struct ChangedPosition<T> {
    position_idx: usize,
    old_value: T,
}

#[cfg(feature = "monte_carlo")]
#[derive(Clone, Copy)]
pub enum ContainerOption<T> {
    This(T),
    Other {
        container_idx: usize,
        changed_position: T,
    },
}

pub mod adder {
    use std::ops::Add;

    pub struct SyncAddError;

    pub trait SyncAdderSender<T: Add<Output = T>> {
        fn send(&mut self, value: T) -> Result<(), SyncAddError>;
    }
    pub trait SyncAdderReciever<T> {
        fn recieve(&mut self) -> Result<T, SyncAddError>;
    }
}
