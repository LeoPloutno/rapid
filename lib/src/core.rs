use crate::stat::Stat;
use std::{
    borrow::Borrow,
    marker::PhantomData,
    num::NonZeroUsize,
    ops::Deref,
    ptr,
    range::Range,
    slice::{self, ChunksExact},
};

/// Information about atoms of the same type.
#[derive(Clone, Debug)]
pub struct AtomTypeInfo<T> {
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

mod map_in_slice {
    use std::{borrow::Borrow, ops::Deref, ptr, range::Range, slice};

    #[derive(Clone, Copy, Debug)]
    pub struct MapInContiguousSlice<'a, T, U: ?Sized> {
        slice: &'a [T],
        this: &'a U,
    }

    impl<'a, T, U: ?Sized> MapInContiguousSlice<'a, T, U> {
        pub const fn this(&self) -> &U {
            self.this
        }

        pub const fn slice(&self) -> &[T] {
            self.slice
        }
    }

    impl<'a, T> MapInContiguousSlice<'a, T, T> {
        pub const fn before(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.slice;
            }
            let slice_ptr = self.slice.as_ptr();
            let this_ptr = ptr::from_ref(self.this);
            unsafe {
                // SAFETY: - `slice_ptr` is derived from a reference.
                //         - The offset of an element - `self.this` - from the
                //           origin - `self.slice` if always less than the length of the slice.
                slice::from_raw_parts(
                    slice_ptr,
                    // SAFETY: By construction, `self.this` points to an element of `self.slice`,
                    //         so it always exceeds or is the start of the slice.
                    this_ptr.offset_from_unsigned(slice_ptr),
                )
            }
        }

        pub const fn after(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.slice;
            }
            let slice_end_ptr = self.slice.as_ptr_range().end;
            let this_ptr = ptr::from_ref(self.this);
            unsafe {
                // SAFETY: - By construction, `self.this` points to an element of `self.slice`.
                //         - `this_ptr + (slice_end_ptr - this_ptr) = slice_end_ptr`.
                slice::from_raw_parts(
                    this_ptr,
                    // SAFETY: By construction, `self.this` points to an element of `self.slice`,
                    //         so the it does not exceed the end of the slice.
                    slice_end_ptr.offset_from_unsigned(this_ptr),
                )
            }
        }

        pub const fn element_offset(&self) -> usize {
            if const { size_of::<T>() == 0 } {
                panic!("elements are zero-sized");
            }
            // SAFETY: By construction, `self.this` points to an element of `self.slice`,
            //         so the it does not exceed the end of the slice.
            unsafe { ptr::from_ref(self.this).offset_from_unsigned(self.slice.as_ptr()) }
        }
    }

    impl<'a, T> MapInContiguousSlice<'a, T, [T]> {
        pub const fn before(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.slice;
            }
            let slice_ptr = self.slice.as_ptr();
            let this_ptr = self.this.as_ptr();
            unsafe {
                // SAFETY: - `slice_ptr` is derived from a reference.
                //         - The offset of a subslice - `self.this` from the
                //           origin - `self.slice` - is always less than or equal to the
                //           length of the slice.
                slice::from_raw_parts(
                    slice_ptr,
                    // SAFETY: By construction, `self.this` points to a subslice entirely within
                    //         `self.slice`, so its start always exceeds or is the slice's.
                    this_ptr.offset_from_unsigned(slice_ptr),
                )
            }
        }

        pub const fn after(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.slice;
            }
            let slice_end_ptr = self.slice.as_ptr_range().end;
            let this_end_ptr = self.this.as_ptr_range().end;
            unsafe {
                // SAFETY: - By construction, `self.this` points to a subslice entirely within
                //           `self.slice`. Thus, the end of said subslice also points within it.
                //         - `this_end_ptr + (slice_end_ptr - this_end_ptr) = slice_end_ptr`.
                slice::from_raw_parts(
                    this_end_ptr,
                    // SAFETY: By construction, `self.this` points to a subslice entirely within
                    //         `self.slice`, so the its end does not exceed the slice's.
                    slice_end_ptr.offset_from_unsigned(this_end_ptr),
                )
            }
        }

        pub const fn subslice_range(&self) -> Range<usize> {
            if const { size_of::<T>() == 0 } {
                panic!("elements are zero-sized");
            }
            let this_len = self.this.len();
            unsafe {
                // SAFETY: By construction, `self.this` points to a subslice entirely within `self.slice`,
                //         so its start does not preceed the slice's.
                let start = self.this.as_ptr().offset_from_unsigned(self.slice.as_ptr());
                Range {
                    start,
                    // SAFETY: Adding the length of a subslice to its start cannot overflow.
                    end: start.unchecked_add(this_len),
                }
            }
        }
    }

    impl<'a, T, U: ?Sized> Deref for MapInContiguousSlice<'a, T, U> {
        type Target = U;

        /// Equivalent to [`MapInContiguousSlice::this`].
        fn deref(&self) -> &U {
            self.this
        }
    }

    impl<'a, T, U: ?Sized> Borrow<U> for MapInContiguousSlice<'a, T, U> {
        /// Equivalent to [`MapInContiguousSlice::this`].
        fn borrow(&self) -> &U {
            self.this
        }
    }

    impl<'a, T, U: ?Sized> AsRef<U> for MapInContiguousSlice<'a, T, U> {
        /// Equivalent to [`MapInContiguousSlice::this`].
        fn as_ref(&self) -> &U {
            self.this
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct MapInDiscontiguousSlice<'a, T, U: ?Sized> {
        before: &'a [T],
        this: &'a U,
        after: &'a [T],
    }

    impl<'a, T> MapInDiscontiguousSlice<'a, T, [T]> {
        pub const fn this(&self) -> &[T] {
            self.this
        }

        pub const fn before(&self) -> &[T] {
            self.before
        }

        pub const fn after(&self) -> &[T] {
            self.after
        }
    }

    impl<'a, T, U: ?Sized> Deref for MapInDiscontiguousSlice<'a, T, U> {
        type Target = U;

        /// Equivalent to [`MapInDiscontiguousSlice::this`].
        fn deref(&self) -> &U {
            self.this
        }
    }

    impl<'a, T, U: ?Sized> Borrow<U> for MapInDiscontiguousSlice<'a, T, U> {
        /// Equivalent to [`MapInDiscontiguousSlice::this`].
        fn borrow(&self) -> &U {
            self.this
        }
    }

    impl<'a, T, U: ?Sized> AsRef<U> for MapInDiscontiguousSlice<'a, T, U> {
        /// Equivalent to [`MapInDiscontiguousSlice::this`].
        fn as_ref(&self) -> &U {
            self.this
        }
    }

    pub type ElementInContiguousSlice<'a, T> = MapInContiguousSlice<'a, T, T>;

    pub type SliceInContiguousSlice<'a, T> = MapInContiguousSlice<'a, T, [T]>;

    pub type ElementInDiscontiguousSlice<'a, T> = MapInDiscontiguousSlice<'a, T, T>;

    pub type SliceInDiscontinuousSlice<'a, T> = MapInDiscontiguousSlice<'a, T, [T]>;
}
pub use map_in_slice::{
    ElementInContiguousSlice, ElementInDiscontiguousSlice, MapInContiguousSlice, MapInDiscontiguousSlice,
    SliceInContiguousSlice, SliceInDiscontinuousSlice,
};

pub struct AtomType<V>(V);

pub struct ImageInSystem<'a, V>(SliceInContiguousSlice<'a, AtomType<V>>);

pub struct ImagesIter<'a, V> {
    start: *const AtomType<V>,
    end: *const AtomType<V>,
    image_size: NonZeroUsize,
    phantom: PhantomData<&'a [AtomType<V>]>,
}

impl<'a, V> ImageInSystem<'a, V> {
    pub const fn this(&self) -> &[AtomType<V>] {
        self.0.this()
    }

    pub const fn before(&self) -> ImagesIter<'a, V> {
        let before = self.0.before();
        let image_size = self.this().len();
        assert!(image_size > 0);
        // SAFETY: Checked above that `image_size > 0`.
        let image_size = unsafe { NonZeroUsize::new_unchecked(image_size) };
        assert!(matches!(before.len().checked_rem(image_size.get()), Some(0)));
        let before_range = before.as_ptr_range();
        ImagesIter {
            start: before_range.start,
            end: before_range.end,
            image_size,
            phantom: PhantomData,
        }
    }

    pub const fn after(&self) -> ImagesIter<'a, V> {
        let after = self.0.after();
        let image_size = self.this().len();
        assert!(image_size > 0);
        // SAFETY: Checked above that `image_size > 0`.
        let image_size = unsafe { NonZeroUsize::new_unchecked(image_size) };
        assert!(matches!(after.len().checked_rem(image_size.get()), Some(0)));
        let after_range = after.as_ptr_range();
        ImagesIter {
            start: after_range.start,
            end: after_range.end,
            image_size,
            phantom: PhantomData,
        }
    }
}

pub struct GroupInType<'a, V>(SliceIn)
//! Core functionalities used throughout the whole project.

use std::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Range, Sub, SubAssign},
    sync::LockResult,
};

use arc_rw_lock::{MappedRwLockReadWholeGuard, MappedRwLockWriteGuard, UniqueArcElementRwLock, UniqueArcSliceRwLock};

mod atoms;

pub use atoms::{AtomType, GroupSizes, GroupSizesIter, GroupsIter};

pub mod error;

pub mod marker {
    //! Marker traits for allowing default implementations.

    /// A marker trait for types that can implement `Leading[...]`
    /// traits by reusing their `Inner[...]` implementation.
    pub trait InnerIsLeading {}

    /// A marker trait for types that can implement `Trailing[...]`
    /// traits by reusing their `Inner[...]` implementation.
    pub trait InnerIsTrailing {}
}

pub mod stat;

pub mod sync_ops;

pub mod factory;

/// A macro that allows pattern-matching items of [zipped iterators](zip_iterators).
#[macro_export]
macro_rules! zip_items {
    ($item1:pat, $item2:pat $(,)?) => {
        ($item1, $item2)
    };
    ($item:pat, $($items:pat),+ $(,)?) => {
        ($item, zip_items!($($items),+))
    };
}
pub use zip_items;

/// A macro that automatically zips all provided iterators in a consistent manner.
#[macro_export]
macro_rules! zip_iterators {
    ($iter:expr) => {
        $iter
    };
    ($iter1:expr, $iter2:expr $(,)?) => {
        $iter1.into_iter().zip($iter2)
    };
    ($iter:expr, $($iters:expr),+ $(,)?) => {
        $iter.into_iter().zip(zip_iterators!($($iters),+))
    };
}
pub use zip_iterators;

/// A trait for objects that can be used as vectors.
pub trait Vector<const N: usize>:
    Sized
    + From<[Self::Element; N]>
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

    /// Returns the index of the group.
    pub fn index(&self) -> usize {
        self.0.as_ref().element_offset()
    }
}

/// Exchange potential expansion scheme.
#[derive(Clone, Copy, Debug)]
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

/// A wrapper for implementors of `Decoupled` traits.
pub struct Decoupled<T: ?Sized>(pub(crate) T);

impl<T> Decoupled<T> {
    /// Wraps the provided value with `Decoupled`.
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}

/// A wrapper for implementors of `Additive` traits.
pub struct Additive<T: ?Sized>(pub(crate) T);

impl<T> Additive<T> {
    /// Wraps the provided value with `Additive`.
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}

/// A wrapper for implementors of `Multiplicative` traits.
pub struct Multiplicative<T: ?Sized>(pub(crate) T);

impl<T> Multiplicative<T> {
    /// Wraps the provided value with `Multiplicative`.
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}
