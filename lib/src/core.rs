use std::{ops::Range, sync::LockResult};

use arc_rw_lock::{MappedRwLockReadWholeGuard, MappedRwLockWriteGuard, UniqueArcElementRwLock, UniqueArcSliceRwLock};

/// Information about atoms of the same type.
#[derive(Clone, Debug)]
pub struct AtomType<T> {
    /// Unique identifier.
    pub id: usize,
    /// The range of indices of groups allocated to this type.
    pub span: Range<usize>,
    /// The mass of a single atom of this type.
    pub mass: T,
    /// Atomic symbol
    pub label: String,
}

pub struct GroupTypeHandle<T>(UniqueArcSliceRwLock<T>);

impl<T> GroupTypeHandle<T> {
    pub fn read(&self) -> &[T] {
        self.0.as_ref().read()
    }

    pub fn write(&mut self) -> MappedRwLockWriteGuard<'_, [T]> {
        self.0.as_mut().write()
    }

    pub fn read_type(&self) -> LockResult<MappedRwLockReadWholeGuard<'_, [T]>> {
        self.0.as_ref().read_whole()
    }

    pub fn range(&self) -> Range<usize> {
        self.0.as_ref().subslice_range()
    }
}

pub struct GroupImageHandle<G>(UniqueArcElementRwLock<G>);

impl<G> GroupImageHandle<G> {
    pub fn read(&self) -> &G {
        self.0.as_ref().read()
    }

    pub fn write(&mut self) -> MappedRwLockWriteGuard<'_, G> {
        self.0.as_mut().write()
    }

    pub fn read_image(&self) -> LockResult<MappedRwLockReadWholeGuard<[G]>> {
        self.0.as_ref().read_whole()
    }

    pub fn index(&self) -> usize {
        self.0.as_ref().element_offset()
    }
}

#[derive(Clone, Debug)]
pub struct CommError<T> {
    pub replica: usize,
    pub group: AtomType<T>,
}

pub trait Factory<'a, T> {
    type Leading: 'a + ?Sized;
    type Inner: 'a + ?Sized;
    type Trailing: 'a + ?Sized;
    type LeadingIt: Iterator<Item = &'a Self::Leading>;
    type InnerIt: Iterator<Item: Iterator<Item = &'a Self::Inner>>;
    type TrailingIt: Iterator<Item = &'a Self::Trailing>;

    fn produce(
        &'a mut self,
        inner_replicas: usize,
        groups: &[AtomType<T>],
    ) -> (Self::LeadingIt, Self::InnerIt, Self::TrailingIt);
}

pub trait FullFactory<'a, T> {
    type Main: 'a;
    type Leading: 'a + ?Sized;
    type Inner: 'a + ?Sized;
    type Trailing: 'a + ?Sized;
    type LeadingIt: Iterator<Item = &'a Self::Leading>;
    type InnerIt: Iterator<Item: Iterator<Item = &'a Self::Inner>>;
    type TrailingIt: Iterator<Item = &'a Self::Trailing>;

    fn produce(
        &'a mut self,
        inner_replicas: usize,
        groups: &[AtomType<T>],
    ) -> (Self::Main, Self::LeadingIt, Self::InnerIt, Self::TrailingIt);
}

pub trait GroupRecord {
    fn group_index(&self) -> usize;
}
