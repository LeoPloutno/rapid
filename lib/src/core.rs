use std::range::Range;

/// Information about atoms of the same type.
#[derive(Clone, Copy, Debug)]
pub struct AtomGroupInfo<T> {
    /// Unique identifier.
    pub id: usize,
    /// The range of indices corresponding to this group.
    pub span: Range<usize>,
    /// The mass of a single atom of this group.
    pub mass: T,
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
