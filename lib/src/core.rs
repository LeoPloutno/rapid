use std::range::Range;

/// Information about atoms of the same type.
#[derive(Clone, Debug)]
pub struct AtomGroupInfo<T> {
    /// Unique identifier.
    pub id: usize,
    /// The range of indices corresponding to this group.
    pub span: Range<usize>,
    /// The mass of a single atom of this group.
    pub mass: T,
    /// Atomic symbol
    pub label: String,
}

pub struct CommError {
    pub replica: usize,
    pub group_idx: usize,
}
