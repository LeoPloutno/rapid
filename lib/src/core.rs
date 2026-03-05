//! Core functionalities used throughout the whole project.

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Range, Sub, SubAssign},
    sync::LockResult,
};

use arc_rw_lock::{MappedRwLockReadWholeGuard, MappedRwLockWriteGuard, UniqueArcElementRwLock, UniqueArcSliceRwLock};

use crate::core::stat::Stat;

pub mod marker {
    //! Marker traits for allowing default implementations.

    /// A marker trait for types that can implement `Leading[...]`
    /// traits by reusing their `Inner[...]` implementation.
    pub trait InnerIsLeading {}

    /// A marker trait for types that can implement `Trailing[...]`
    /// traits by reusing their `Inner[...]` implementation.
    pub trait InnerIsTrailing {}
}

pub mod stat {
    //! Types and traits meant to distinguish between different types
    //! of ensemble statistics.

    use std::ops::{Deref, DerefMut};

    /// An enum differentiating between distinguishable and bosonic statistics.
    #[derive(Clone, Copy, Debug)]
    #[non_exhaustive]
    pub enum Stat<D, B> {
        /// Distinguishable statistics.
        Distinguishable(D),
        /// Bosonic statistics.
        Bosonic(B),
    }

    impl<D, B> Stat<D, B> {
        /// Converts from `Stat<D, B>` to
        /// `Stat<&D::Target, &B::Target>`.
        ///
        /// Leaves the original `Stat` in-place,
        /// creating a new one containing references to the inner types' `Deref::Target` types.
        pub fn as_deref(&self) -> Stat<&<D as Deref>::Target, &<B as Deref>::Target>
        where
            D: Deref,
            B: Deref,
        {
            match self {
                Self::Distinguishable(dist) => Stat::Distinguishable(dist),
                Self::Bosonic(boson) => Stat::Bosonic(boson),
            }
        }

        /// Converts from `Stat<D, B>` to
        /// `Stat<&mut D::Target, &mut B::Target>`.
        ///
        /// Leaves the original `Stat` in-place,
        /// creating a new one containing mutable references to the inner types' `Deref::Target` types.
        pub fn as_deref_mut(&mut self) -> Stat<&mut <D as Deref>::Target, &mut <B as Deref>::Target>
        where
            D: DerefMut,
            B: DerefMut,
        {
            match self {
                Self::Distinguishable(dist) => Stat::Distinguishable(dist),
                Self::Bosonic(boson) => Stat::Bosonic(boson),
            }
        }
    }

    /// A trait for marking exchange potentials of distinguishable particles.
    pub trait Distinguishable {}

    /// A trait for marking exchange potentials of bosons.
    pub trait Bosonic {}
}

pub mod sync_ops {
    //! Traits for parallelized calculations.

    /// A trait for objects which add up values and send the sum to a `SyncAddReciever`.
    pub trait SyncAddSender<T> {
        /// The type associated with an error returned by the implementor.
        type Error;

        /// Sends `value` to the adder.
        fn send(&mut self, value: T) -> Result<(), Self::Error>;

        /// Sends an empty message to the adder.
        fn send_empty(&mut self) -> Result<(), Self::Error>;
    }

    /// A trait for objects which recieve the sum calculated by `SyncAddSender`s.
    pub trait SyncAddReciever<T> {
        /// The type associated with an error returned by the implementor.
        type Error;

        /// Recieves the sum of all non-empty messages.
        fn recieve_sum(&mut self) -> Result<Option<T>, Self::Error>;
    }

    /// A trait for objects which multiply values and send the product to a `SyncAddReciever`.
    pub trait SyncMulSender<T> {
        /// The type associated with an error returned by the implementor.
        type Error;

        /// Sends `value` to the multiplier.
        fn send(&mut self, value: T) -> Result<(), Self::Error>;

        /// Sends an empty message to the multiplier.
        fn send_empty(&mut self) -> Result<(), Self::Error>;
    }

    /// A trait for objects which recieve the product calculated by `SyncAddSender`s.
    pub trait SyncMulReciever<T> {
        /// The type associated with an error returned by the implementor.
        type Error;

        /// Recieves the product of all non-empty messages.
        fn recieve_prod(&mut self) -> Result<Option<T>, Self::Error>;
    }
}

pub mod factory {
    //! Traits for producing different yet connected types of objects.
    use crate::core::AtomType;

    /// A trait for "factories" that produce iterators for leading, inner and trailing images.
    pub trait Factory<'a, T> {
        /// The object used in a leading thread.
        type Leading: 'a;
        /// The object used in an inner thread.
        type Inner: 'a;
        /// The object used in a trailing thread.
        type Trailing: 'a;
        /// The iterator producing `Leading` objects.
        type LeadingIter: ExactSizeIterator<Item = Self::Leading>;
        /// The iterator producing iterators that produce `Inner` objects.
        type InnerIter: ExactSizeIterator<Item: ExactSizeIterator<Item = Self::Inner>>;
        /// The iterator producing `Trailing` objects.
        type TrailingIter: ExactSizeIterator<Item = Self::Trailing>;

        /// Produces the iterators.
        fn produce(
            &'a mut self,
            inner_images: usize,
            atom_types: &[AtomType<T>],
            groups_sizes: &[usize],
        ) -> (Self::LeadingIter, Self::InnerIter, Self::TrailingIter);
    }

    /// A trait for "factories" that produce iterators for the main thread and leading, inner and trailing images.
    pub trait FullFactory<'a, T> {
        /// The object used in the main thread.
        type Main: 'a;
        /// The object used in a leading thread.
        type Leading: 'a;
        /// The object used in an inner thread.
        type Inner: 'a;
        /// The object used in a trailing thread.
        type Trailing: 'a;
        /// The iterator producing `Leading` objects.
        type LeadingIter: ExactSizeIterator<Item = Self::Leading>;
        /// The iterator producing iterators that produce `Inner` objects.
        type InnerIter: ExactSizeIterator<Item: ExactSizeIterator<Item = Self::Inner>>;
        /// The iterator producing `Trailing` objects.
        type TrailingIter: ExactSizeIterator<Item = Self::Trailing>;

        /// Produces the main object and the iterators.
        fn produce(
            &'a mut self,
            inner_images: usize,
            atom_types: &[AtomType<T>],
            groups_sizes: &[usize],
        ) -> (Self::Main, Self::LeadingIter, Self::InnerIter, Self::TrailingIter);
    }
}

/// A trait for objects that can be used as vectors.
pub trait Vector<const N: usize>:
    Sized
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Self::Element, Output = Self>
    + MulAssign<Self::Element>
    + Div<Self::Element, Output = Self>
    + DivAssign<Self::Element>
    + Neg<Output = Self>
{
    /// The type of the element of the vector.
    type Element;

    /// Converts to a reference to an array.
    fn as_array(&self) -> &[Self::Element; N];

    /// Converts to a mutable reference to an array.
    fn as_mut_array(&mut self) -> &mut [Self::Element; N];

    /// Calculates the square of the magnitude (length) of the vector.
    fn magnitude_squared(self) -> Self::Element;

    /// Calculates the dot product of `self` with `rhs`.
    fn dot(self, rhs: Self) -> Self::Element;
}

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

/// A lock which grants mutable access to a single group
/// or immutable access to all groups of the same type.
pub struct GroupTypeHandle<T>(UniqueArcSliceRwLock<T>);

impl<T> GroupTypeHandle<T> {
    /// Grants immutable access to the underlying group.
    pub fn read(&self) -> &[T] {
        self.0.as_ref().read()
    }

    /// Grants mutable access to the underlying group.
    pub fn write(&mut self) -> MappedRwLockWriteGuard<'_, [T]> {
        self.0.as_mut().write()
    }

    /// Grants immutable access to all groups allocated to a type of atoms,
    /// returning an error if a thread holding a mutable guard panicked.
    pub fn read_type(&self) -> LockResult<MappedRwLockReadWholeGuard<'_, [T]>> {
        self.0.as_ref().read_whole()
    }

    /// Returns the range of the subslice the underlying group
    /// represents amongst all atoms of the same type.
    pub fn range(&self) -> Range<usize> {
        self.0.as_ref().subslice_range()
    }
}

/// A lock which grants mutable access to a single image
/// or immutable access to all images.
pub struct GroupImageHandle<G>(UniqueArcElementRwLock<G>);

impl<G> GroupImageHandle<G> {
    /// Grants immutable access to the underlying image.
    pub fn read(&self) -> &G {
        self.0.as_ref().read()
    }

    /// Grants immutable access to the underlying image.
    pub fn write(&mut self) -> MappedRwLockWriteGuard<'_, G> {
        self.0.as_mut().write()
    }

    /// Grants immutable access to images,
    /// returning an error if a thread holding a mutable guard panicked.
    pub fn read_image(&self) -> LockResult<MappedRwLockReadWholeGuard<'_, [G]>> {
        self.0.as_ref().read_whole()
    }

    /// Returns the index of the underlying image amongst all images.
    pub fn index(&self) -> usize {
        self.0.as_ref().element_offset()
    }
}

/// A miscellaneous error used by [`run`].
///
/// [`run`]: crate::run
#[derive(Clone, Debug)]
pub enum CommError {
    /// The error arose in the main thread.
    Main,
    /// The error arose in a leading thread.
    Leading {
        /// The index of the group the thread is assigned to.
        group: usize,
    },
    /// The error arose in an inner thread.
    Inner {
        /// The image the thread is assigned to.
        image: usize,
        /// The index of the group the thread is assigned to.
        group: usize,
    },
    /// The error arose in a trailing thread.
    Trailing {
        /// The index of the group the thread is assigmed to.
        group: usize,
    },
}

impl Display for CommError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Main => write!(f, "Something happened in the main thread"),
            Self::Leading { group } => write!(
                f,
                "Something happened in a thread dedicated to group #{} in the first image",
                group
            ),
            Self::Inner { image, group } => write!(
                f,
                "Something happened in a thread dedicated to group #{} in image #{}",
                group, image
            ),
            Self::Trailing { group } => write!(
                f,
                "Something happened in a thread dedicated to group #{} in the last image",
                group
            ),
        }
    }
}

impl Error for CommError {}

/// A trait for objects that "know" which group they are responsible for.
pub trait GroupRecord {
    /// Returns the index of the group which is under the jurisdiction of the implementor.
    fn group_index(&self) -> usize;
}

/// Exchange potential expansion scheme.
pub enum Scheme<T, U> {
    /// Regular, unexpanded.
    Regular(T),
    /// Expanded to the second order.
    QuadraticExpansion(U),
}

/// A struct which contains the two objects dependent on the expansion
/// scheme - the propagator and the exchange potential.
pub struct SchemeDependent<Prop, ExchPot> {
    /// The propagator.
    pub propagator: Prop,
    /// The exchange potential.
    pub exchange_potential: ExchPot,
}
