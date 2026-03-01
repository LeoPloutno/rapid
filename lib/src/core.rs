use std::{ops::Range, sync::LockResult};

use arc_rw_lock::{MappedRwLockReadWholeGuard, MappedRwLockWriteGuard, UniqueArcElementRwLock, UniqueArcSliceRwLock};

use crate::stat::Stat;

/// Information about atoms of the same type.
#[derive(Clone, Debug)]
pub struct AtomType<T> {
    /// Unique identifier.
    pub id: usize,
    /// Atomic symbol
    pub label: String,
    /// The range of indices of groups allocated to this type.
    pub groups: Range<usize>,
    /// The mass of a single atom of this type.
    pub mass: T,
    /// Whether the atoms are distinguishable.
    pub statistic: Stat<(), ()>,
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

    pub fn read_image(&self) -> LockResult<MappedRwLockReadWholeGuard<'_, [G]>> {
        self.0.as_ref().read_whole()
    }

    pub fn index(&self) -> usize {
        self.0.as_ref().element_offset()
    }
}

#[derive(Clone, Debug)]
pub enum CommError {
    Main,
    Leading { group: usize },
    Inner { image: usize, group: usize },
    Trailing { group: usize },
}

pub trait Factory<'a, T> {
    type Leading: 'a;
    type Inner: 'a;
    type Trailing: 'a;
    type LeadingIter: ExactSizeIterator<Item = Self::Leading>;
    type InnerIter: ExactSizeIterator<Item: ExactSizeIterator<Item = Self::Inner>>;
    type TrailingIter: ExactSizeIterator<Item = Self::Trailing>;

    fn produce(
        &'a mut self,
        inner_images: usize,
        atom_types: &[AtomType<T>],
        groups_sizes: &[usize],
    ) -> (Self::LeadingIter, Self::InnerIter, Self::TrailingIter);
}

pub trait FullFactory<'a, T> {
    type Main: 'a;
    type Leading: 'a;
    type Inner: 'a;
    type Trailing: 'a;
    type LeadingIter: ExactSizeIterator<Item = Self::Leading>;
    type InnerIter: ExactSizeIterator<Item: ExactSizeIterator<Item = Self::Inner>>;
    type TrailingIter: ExactSizeIterator<Item = Self::Trailing>;

    fn produce(
        &'a mut self,
        inner_images: usize,
        atom_types: &[AtomType<T>],
        groups_sizes: &[usize],
    ) -> (Self::Main, Self::LeadingIter, Self::InnerIter, Self::TrailingIter);
}

pub trait GroupRecord {
    fn group_index(&self) -> usize;
}
