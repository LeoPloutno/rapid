use std::{iter::FusedIterator, num::NonZeroUsize, slice::Iter};

use crate::core::stat::Stat;

/// Information about atoms of the same type.
#[derive(Clone, Debug)]
pub struct AtomTypeInfo<T> {
    /// Unique identifier.
    pub id: usize,
    /// Atomic symbol
    pub label: String,
    /// The sizes of the groups this type is split into.
    pub groups: GroupSizes,
    /// The mass of a single atom of this type.
    pub mass: T,
    /// Whether the atoms are distinguishable.
    pub statistic: Stat<(), ()>,
}

/// A struct containig information about the sizes of
/// the groups a type is split into.
#[derive(Clone, Copy, Debug)]
pub struct GroupSizes {
    total: NonZeroUsize,
    groups: NonZeroUsize,
}

impl GroupSizes {
    /// Returns an iterator over the sizes of the groups.
    pub fn iter(&self) -> GroupSizesIter {
        debug_assert!(self.total > self.groups);

        let small_group_size = usize::from(self.total) / self.groups;
        let large_groups = usize::from(self.total) % small_group_size;
        GroupSizesIter {
            small_groups: usize::from(self.groups) - large_groups,
            large_groups,
            small_group_size,
        }
    }

    /// Returns the total number of atoms.
    pub fn total(&self) -> usize {
        self.total.into()
    }

    /// Returns the number of groups.
    pub fn groups(&self) -> usize {
        self.groups.into()
    }
}

/// An iterator over sizes of groups of atoms of the same type.
#[derive(Clone, Copy, Debug)]
pub struct GroupSizesIter {
    small_groups: usize,
    large_groups: usize,
    small_group_size: usize,
}

impl Iterator for GroupSizesIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.small_groups > 0 {
            self.small_groups -= 1;
            Some(self.small_group_size)
        } else if self.large_groups > 0 {
            self.large_groups -= 1;
            Some(self.small_group_size + 1)
        } else {
            None
        }
    }

    fn min(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        if self.small_groups > 0 {
            Some(self.small_group_size)
        } else if self.large_groups > 0 {
            Some(self.small_group_size + 1)
        } else {
            None
        }
    }

    fn max(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        if self.large_groups > 0 {
            Some(self.small_group_size + 1)
        } else if self.small_groups > 0 {
            Some(self.small_group_size)
        } else {
            None
        }
    }
}

impl DoubleEndedIterator for GroupSizesIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.large_groups > 0 {
            self.large_groups -= 1;
            Some(self.small_group_size + 1)
        } else if self.small_groups > 0 {
            self.small_groups -= 1;
            Some(self.small_group_size)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for GroupSizesIter {
    fn len(&self) -> usize {
        self.small_groups + self.large_groups
    }
}

impl FusedIterator for GroupSizesIter {}

/// An iterator over `(atom_type, group_size)` pairs.
pub struct GroupsIter<'a, T> {
    atom_types_iter: Iter<'a, AtomType<T>>,
    opt_group_sizes_iter: Option<(&'a AtomType<T>, GroupSizesIter)>,
}

impl<'a, T> GroupsIter<'a, T> {
    /// Returns a `GroupsIter`.
    pub fn from_atom_types(atom_types: &'a [AtomType<T>]) -> GroupsIter<'a, T> {
        let mut atom_types_iter = atom_types.iter();
        let opt_group_sizes_iter = atom_types_iter
            .next()
            .map(|atom_type| (atom_type, atom_type.groups.iter()));
        Self {
            atom_types_iter,
            opt_group_sizes_iter,
        }
    }
}

impl<'a, T> Iterator for GroupsIter<'a, T> {
    type Item = (&'a AtomType<T>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((atom_type, iter)) = &mut self.opt_group_sizes_iter {
            match iter.next() {
                Some(group_size) => Some((atom_type, group_size)),
                None => match self.atom_types_iter.next() {
                    Some(atom_type) => {
                        let mut iter_new = atom_type.groups.iter();
                        match iter_new.next() {
                            Some(group_size) => {
                                self.opt_group_sizes_iter = Some((atom_type, iter_new));
                                Some((atom_type, group_size))
                            }
                            None => {
                                self.opt_group_sizes_iter = None;
                                None
                            }
                        }
                    }
                    None => {
                        self.opt_group_sizes_iter = None;
                        None
                    }
                },
            }
        } else {
            None
        }
    }
}

impl<'a, T> FusedIterator for GroupsIter<'a, T> {}
