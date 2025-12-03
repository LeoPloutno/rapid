/// Information about atoms of the same type.
#[derive(Clone, Copy, Debug)]
pub struct AtomGroupInfo<T> {
    /// Unique identifier.
    pub id: usize,
    /// The mass of a single atom in this group.
    pub mass: T,
}

#[cfg(feature = "monte_carlo")]
#[derive(Clone, Copy)]
pub struct ChangedPosition<T> {
    group_idx: usize,
    position_idx: usize,
    old_value: T,
}

#[cfg(feature = "monte_carlo")]
#[derive(Clone, Copy)]
pub enum GroupOption<T> {
    This(T),
    Other { group_idx: usize, value: T },
}

#[cfg(feature = "monte_carlo")]
#[derive(Clone, Copy)]
pub enum ReplicaOption<T> {
    This(T),
    Prev(T),
    Next(T),
    Other { replica_idx: usize, value: T },
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

pub mod marker {
    pub trait LeadingIsInner {}
    pub trait TrailingIsInner {}
}

pub mod stats {
    pub trait Indistinguishable {}

    pub trait Bosonic {}

    impl<T: ?Sized + Bosonic> Indistinguishable for T {}
}
