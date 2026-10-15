use super::InnerArc;
use crate::{
    ArcReaderLock, ReaderLock,
    lock::{InnerRwLock, MappedRwLock, PoisonLock},
    slice::{ChunksExact, Iter},
};
use std::{
    alloc::{AllocError, Allocator, AllocatorClone, Global, Layout, LayoutError, handle_alloc_error},
    borrow::Borrow,
    convert::AsRef,
    marker::Unsize,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    ops::{CoerceUnsized, Deref, OneSidedRange, OneSidedRangeBound},
    process,
    ptr::{NonNull, Pointee},
    sync::atomic::{self, AtomicUsize, Ordering},
};

pub struct ArcMappedRwLock<T: ?Sized, U: ?Sized = dyn Send + Sync + 'static, A: Allocator = Global> {
    pub(crate) lock: MappedRwLock<T>,
    pub(crate) metadata: <U as Pointee>::Metadata,
    pub(crate) allocator: A,
}

impl<T: ?Sized, U: ?Sized, A: Allocator> ArcMappedRwLock<T, U, A> {
    /// Creates a new instance of `ArcMappedRwLock<T, U, A>` from raw parts.
    ///
    /// # Safety
    ///
    /// - `metadata` is the same as the metadata of the
    ///   `InnerArc<U>` from which `lock` and `data` are derived.
    /// - The underlying allocation has been allocated using `allocator` or a valid clone of its.
    /// - See [`MappedRwLock::from_raw_parts`]
    #[inline]
    pub(crate) const unsafe fn from_raw_parts(
        lock: NonNull<PoisonLock>,
        data: NonNull<T>,
        metadata: <U as Pointee>::Metadata,
        allocator: A,
    ) -> Self {
        Self {
            lock: unsafe { MappedRwLock::from_raw_parts(lock, data) },
            metadata,
            allocator,
        }
    }

    pub const fn allocator(this: &Self) -> &A {
        &this.allocator
    }
}

impl<T> ArcMappedRwLock<T, T> {
    pub fn new(data: T) -> (Self, ArcReaderLock<T>) {
        Self::new_in(data, Global)
    }

    pub fn new_uninit() -> (
        ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
        ArcReaderLock<MaybeUninit<T>>,
    ) {
        Self::new_uninit_in(Global)
    }

    pub fn new_zeroed() -> (
        ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
        ArcReaderLock<MaybeUninit<T>>,
    ) {
        Self::new_zeroed_in(Global)
    }

    pub fn try_new(data: T) -> Result<(Self, ArcReaderLock<T>), AllocError> {
        Self::try_new_in(data, Global)
    }

    pub fn try_new_uninit() -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
            ArcReaderLock<MaybeUninit<T>>,
        ),
        AllocError,
    > {
        Self::try_new_uninit_in(Global)
    }

    pub fn try_new_zeroed() -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
            ArcReaderLock<MaybeUninit<T>>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_in(Global)
    }
}

impl<T, A> ArcMappedRwLock<T, T, A>
where
    A: AllocatorClone,
{
    #[inline]
    fn try_new_in_with_layout(data: T, alloc: A) -> Result<(Self, ArcReaderLock<T, A>), (Layout, AllocError)> {
        let inner = InnerArc {
            counter: AtomicUsize::new(InnerArc::<MaybeUninit<T>>::SHARED_COUNTER_ONE),
            lock: InnerRwLock {
                poison_lock: PoisonLock::new(),
                data,
            },
        };
        let layout = Layout::for_value(&inner);
        let allocation = match alloc.allocate(layout) {
            Ok(ptr) => ptr.cast(),
            Err(err) => return Err((layout, err)),
        };
        // SAFETY: The pointer points to a live and valid allocation tailored to `inner`.
        unsafe {
            allocation.write(inner);
        }
        Ok((
            unsafe {
                // SAFETY: - Provided a clone of the allocator, where `A: AllocatorClone`.,
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - Neither `ArcMappedRwLock` nor `ArcReaderLock` provide mutable
                //           access to this lock so no write operations can be performed using it.
                Self::from_raw_parts(
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).lock.poison_lock),
                    NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).lock.data),
                    (),
                    alloc.clone(),
                )
            },
            ArcReaderLock {
                lock: unsafe {
                    // SAFETY: The pointer is derived from a valid and live allocation of `InnerArc`.
                    ReaderLock::from_raw_parts(
                        // SAFETY: A pointer to a field of a live and valid struct is non-null.
                        NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).lock),
                    )
                },
                allocator: alloc,
            },
        ))
    }

    #[inline]
    fn try_new_uninit_in_with_layout(
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
            ArcReaderLock<MaybeUninit<T>, A>,
        ),
        (Layout, AllocError),
    > {
        let layout = Layout::new::<InnerArc<MaybeUninit<T>>>();
        let allocation = match alloc.allocate(layout) {
            Ok(ptr) => ptr.cast::<InnerArc<MaybeUninit<T>>>(),
            Err(err) => return Err((layout, err)),
        };
        let ptr = allocation.as_ptr();
        // SAFETY: The allocation has been allocated previously.
        unsafe {
            (&raw mut (*ptr).counter).write(AtomicUsize::new(InnerArc::<MaybeUninit<T>>::SHARED_COUNTER_ONE));
            (&raw mut (*ptr).lock.poison_lock).write(PoisonLock::new());
        }
        Ok((
            unsafe {
                // SAFETY: - Provided a clone of the allocator, where `A: AllocatorClone`.,
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - Neither `ArcMappedRwLock` nor `ArcReaderLock` provide mutable
                //           access to this lock so no write operations can be performed using it.
                ArcMappedRwLock::from_raw_parts(
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    NonNull::new_unchecked(&raw mut (*ptr).lock.poison_lock),
                    NonNull::new_unchecked(&raw mut (*ptr).lock.data),
                    (),
                    alloc.clone(),
                )
            },
            ArcReaderLock {
                lock: unsafe {
                    // SAFETY: The pointer is derived from a valid and live allocation of `InnerArc`.
                    ReaderLock::from_raw_parts(
                        // SAFETY: A pointer to a field of a live and valid struct is non-null.
                        NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).lock),
                    )
                },
                allocator: alloc,
            },
        ))
    }

    #[inline]
    fn try_new_zeroed_in_with_layout(
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
            ArcReaderLock<MaybeUninit<T>, A>,
        ),
        (Layout, AllocError),
    > {
        let layout = Layout::new::<InnerArc<MaybeUninit<T>>>();
        let allocation = match alloc.allocate_zeroed(layout) {
            Ok(ptr) => ptr.cast::<InnerArc<MaybeUninit<T>>>(),
            Err(err) => return Err((layout, err)),
        };
        let ptr = allocation.as_ptr();
        // SAFETY: The allocation has been allocated previously.
        unsafe {
            (&raw mut (*ptr).counter).write(AtomicUsize::new(InnerArc::<MaybeUninit<T>>::SHARED_COUNTER_ONE));
            (&raw mut (*ptr).lock.poison_lock).write(PoisonLock::new());
        }
        Ok((
            unsafe {
                // SAFETY: - Provided a clone of the allocator, where `A: AllocatorClone`.,
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - Neither `ArcMappedRwLock` nor `ArcReaderLock` provide mutable
                //           access to this lock so no write operations can be performed using it.
                ArcMappedRwLock::from_raw_parts(
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    NonNull::new_unchecked(&raw mut (*ptr).lock.poison_lock),
                    NonNull::new_unchecked(&raw mut (*ptr).lock.data),
                    (),
                    alloc.clone(),
                )
            },
            ArcReaderLock {
                lock: unsafe {
                    // SAFETY: The pointer is derived from a valid and live allocation of `InnerArc`.
                    ReaderLock::from_raw_parts(
                        // SAFETY: A pointer to a field of a live and valid struct is non-null.
                        NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).lock),
                    )
                },
                allocator: alloc,
            },
        ))
    }

    pub fn new_in(data: T, alloc: A) -> (Self, ArcReaderLock<T, A>) {
        match Self::try_new_in_with_layout(data, alloc) {
            Ok(arc) => arc,
            Err((layout, _)) => handle_alloc_error(layout),
        }
    }

    pub fn new_uninit_in(
        alloc: A,
    ) -> (
        ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
        ArcReaderLock<MaybeUninit<T>, A>,
    ) {
        match Self::try_new_uninit_in_with_layout(alloc) {
            Ok(arc) => arc,
            Err((layout, _)) => handle_alloc_error(layout),
        }
    }

    pub fn new_zeroed_in(
        alloc: A,
    ) -> (
        ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
        ArcReaderLock<MaybeUninit<T>, A>,
    ) {
        match Self::try_new_zeroed_in_with_layout(alloc) {
            Ok(arc) => arc,
            Err((layout, _)) => handle_alloc_error(layout),
        }
    }

    pub fn try_new_in(data: T, alloc: A) -> Result<(Self, ArcReaderLock<T, A>), AllocError> {
        Self::try_new_in_with_layout(data, alloc).map_err(|(_, err)| err)
    }

    pub fn try_new_uninit_in(
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
            ArcReaderLock<MaybeUninit<T>, A>,
        ),
        AllocError,
    > {
        Self::try_new_uninit_in_with_layout(alloc).map_err(|(_, err)| err)
    }

    pub fn try_new_zeroed_in(
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
            ArcReaderLock<MaybeUninit<T>, A>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_in_with_layout(alloc).map_err(|(_, err)| err)
    }
}

enum TryAllocError {
    Layout(LayoutError),
    Alloc { layout: Layout, err: AllocError },
}

impl From<LayoutError> for TryAllocError {
    fn from(value: LayoutError) -> Self {
        Self::Layout(value)
    }
}

impl<T> ArcMappedRwLock<[T], [T]> {
    pub fn new_uninit_slice(
        len: usize,
    ) -> (
        ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
        ArcReaderLock<[MaybeUninit<T>]>,
    ) {
        Self::new_uninit_slice_in(len, Global)
    }

    pub fn new_zeroed_slice(
        len: usize,
    ) -> (
        ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
        ArcReaderLock<[MaybeUninit<T>]>,
    ) {
        Self::new_zeroed_slice_in(len, Global)
    }

    pub fn try_new_uninit_slice(
        len: usize,
    ) -> Result<
        (
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
            ArcReaderLock<[MaybeUninit<T>]>,
        ),
        AllocError,
    > {
        Self::try_new_uninit_slice_in(len, Global)
    }

    pub fn try_new_zeroed_slice(
        len: usize,
    ) -> Result<
        (
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
            ArcReaderLock<[MaybeUninit<T>]>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_slice_in(len, Global)
    }
}

impl<T, A> ArcMappedRwLock<[T], [T], A>
where
    A: AllocatorClone,
{
    #[inline]
    fn try_new_uninit_slice_in_with_layout(
        len: usize,
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
            ArcReaderLock<[MaybeUninit<T>], A>,
        ),
        TryAllocError,
    > {
        let layout = InnerArc::<[MaybeUninit<T>]>::get_layout(Layout::array::<MaybeUninit<T>>(len)?)?;
        let allocation = match alloc.allocate(layout) {
            Ok(ptr) => NonNull::<InnerArc<[MaybeUninit<T>]>>::from_raw_parts(ptr.to_raw_parts().0, len),
            Err(err) => return Err(TryAllocError::Alloc { layout, err }),
        };
        let ptr = allocation.as_ptr();
        // SAFETY: The allocation has been allocated previously.
        unsafe {
            (&raw mut (*ptr).counter).write(AtomicUsize::new(1));
            (&raw mut (*ptr).lock.poison_lock).write(PoisonLock::new());
        }
        Ok((
            // ArcMappedRwLock {
            //     lock: MappedRwLock {
            //         lock: inner,
            //         // SAFETY: A pointer to a field of a live and valid struct is non-null.
            //         data: unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock.data) },
            //     },
            //     allocator: alloc.clone(),
            // },
            unsafe {
                // SAFETY: - By construction, `len` is the length of the unsized part of the allocation.
                //         - Provided a clone of the allocator, where `A: AllocatorClone`.,
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - Neither `ArcMappedRwLock` nor `ArcReaderLock` provide mutable
                //           access to this lock so no write operations can be performed using it.
                ArcMappedRwLock::from_raw_parts(
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    NonNull::new_unchecked(&raw mut (*ptr).lock.poison_lock),
                    NonNull::new_unchecked(&raw mut (*ptr).lock.data),
                    len,
                    alloc.clone(),
                )
            },
            ArcReaderLock {
                lock: unsafe {
                    // SAFETY: The pointer is derived from a valid and live allocation of `InnerArc`.
                    ReaderLock::from_raw_parts(
                        // SAFETY: A pointer to a field of a live and valid struct is non-null.
                        NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).lock),
                    )
                },
                allocator: alloc,
            },
        ))
    }

    #[inline]
    fn try_new_zeroed_slice_in_with_layout(
        len: usize,
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
            ArcReaderLock<[MaybeUninit<T>], A>,
        ),
        TryAllocError,
    > {
        let layout = InnerArc::<[MaybeUninit<T>]>::get_layout(Layout::array::<MaybeUninit<T>>(len)?)?;
        let allocation = match alloc.allocate_zeroed(layout) {
            Ok(ptr) => NonNull::<InnerArc<[MaybeUninit<T>]>>::from_raw_parts(ptr.to_raw_parts().0, len),
            Err(err) => return Err(TryAllocError::Alloc { layout, err }),
        };
        let ptr = allocation.as_ptr();
        // SAFETY: The allocation has been allocated previously.
        unsafe {
            (&raw mut (*ptr).counter).write(AtomicUsize::new(1));
            (&raw mut (*ptr).lock.poison_lock).write(PoisonLock::new());
        }
        Ok((
            // ArcMappedRwLock {
            //     lock: MappedRwLock {
            //         lock: inner,
            //         // SAFETY: A pointer to a field of a live and valid struct is non-null.
            //         data: unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock.data) },
            //     },
            //     allocator: alloc.clone(),
            // },
            unsafe {
                // SAFETY: - By construction, `len` is the length of the unsized part of the allocation.
                //         - Provided a clone of the allocator, where `A: AllocatorClone`.,
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - Neither `ArcMappedRwLock` nor `ArcReaderLock` provide mutable
                //           access to this lock so no write operations can be performed using it.
                ArcMappedRwLock::from_raw_parts(
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    NonNull::new_unchecked(&raw mut (*ptr).lock.poison_lock),
                    NonNull::new_unchecked(&raw mut (*ptr).lock.data),
                    len,
                    alloc.clone(),
                )
            },
            ArcReaderLock {
                lock: unsafe {
                    // SAFETY: The pointer is derived from a valid and live allocation of `InnerArc`.
                    ReaderLock::from_raw_parts(
                        // SAFETY: A pointer to a field of a live and valid struct is non-null.
                        NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).lock),
                    )
                },
                allocator: alloc,
            },
        ))
    }

    pub fn new_uninit_slice_in(
        len: usize,
        alloc: A,
    ) -> (
        ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
        ArcReaderLock<[MaybeUninit<T>], A>,
    ) {
        match Self::try_new_uninit_slice_in_with_layout(len, alloc) {
            Ok(arcs) => arcs,
            Err(TryAllocError::Layout(_)) => panic!("computed invalid layout during allocation"),
            Err(TryAllocError::Alloc { layout, .. }) => handle_alloc_error(layout),
        }
    }

    pub fn new_zeroed_slice_in(
        len: usize,
        alloc: A,
    ) -> (
        ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
        ArcReaderLock<[MaybeUninit<T>], A>,
    ) {
        match Self::try_new_zeroed_slice_in_with_layout(len, alloc) {
            Ok(arcs) => arcs,
            Err(TryAllocError::Layout(_)) => panic!("computed invalid layout during allocation"),
            Err(TryAllocError::Alloc { layout, .. }) => handle_alloc_error(layout),
        }
    }

    pub fn try_new_uninit_slice_in(
        len: usize,
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
            ArcReaderLock<[MaybeUninit<T>], A>,
        ),
        AllocError,
    > {
        Self::try_new_uninit_slice_in_with_layout(len, alloc).map_err(|err| match err {
            TryAllocError::Layout(_) => AllocError,
            TryAllocError::Alloc { err, .. } => err,
        })
    }

    pub fn try_new_zeroed_slice_in(
        len: usize,
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
            ArcReaderLock<[MaybeUninit<T>], A>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_slice_in_with_layout(len, alloc).map_err(|err| match err {
            TryAllocError::Layout(_) => AllocError,
            TryAllocError::Alloc { err, .. } => err,
        })
    }
}

impl<T, U, A: Allocator> ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<U>, A> {
    pub const unsafe fn assume_init(self) -> ArcMappedRwLock<T, U, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        ArcMappedRwLock {
            // SAFETY: User-upheld invariant.
            lock: unsafe { lock.assume_init() },
            metadata: (),
            allocator,
        }
    }
}

impl<T, A: Allocator> ArcMappedRwLock<MaybeUninit<T>, [MaybeUninit<T>], A> {
    pub const unsafe fn assume_init(self) -> ArcMappedRwLock<T, [T], A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let metadata = unsafe { (&raw const self.metadata).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        ArcMappedRwLock {
            // SAFETY: User-upheld invariant.
            lock: unsafe { lock.assume_init() },
            metadata,
            allocator,
        }
    }
}

impl<T, A: Allocator> ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A> {
    pub const unsafe fn assume_init(self) -> ArcMappedRwLock<[T], [T], A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let metadata = unsafe { (&raw const self.metadata).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        ArcMappedRwLock {
            // SAFETY: User-upheld invariant.
            lock: unsafe { lock.assume_init() },
            metadata,
            allocator,
        }
    }
}

impl<T, U: ?Sized, A: Allocator> ArcMappedRwLock<[T], U, A> {
    pub fn iter(self) -> Iter<T, U, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        Iter { lock, allocator }
    }

    pub fn chunks_exact(self, chunk_size: usize) -> ChunksExact<T, U, A> {
        assert!(chunk_size != 0, "chunk size must be non-zero");

        // SAFETY: Checked above that `chunk_size != 0`.
        let chunk_size = unsafe { NonZeroUsize::new_unchecked(chunk_size) };

        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let MappedRwLock {
            lock: allocation,
            data: slice,
        } = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);

        let (ptr, len) = slice.to_raw_parts();
        let ptr = ptr.cast::<T>();
        let remainder = len % chunk_size;
        let len_chunks = len - remainder;

        ChunksExact {
            allocation,
            remainder: NonNull::from_raw_parts(
                // SAFETY: By construction, `ptr` points to a slice which contains at least `len_chunks + remainder` elements.
                unsafe { ptr.add(len_chunks) },
                remainder,
            ),
            slice: NonNull::from_raw_parts(ptr, len_chunks),
            chunk_size,
            allocator,
        }
    }
}

impl<T, U: ?Sized, A: AllocatorClone> ArcMappedRwLock<[T], U, A> {
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        match self.split_at_checked(mid) {
            Ok(pair) => pair,
            Err(_) => panic!("mid > len"),
        }
    }

    pub fn split_at_checked(mut self, mid: usize) -> Result<(Self, Self), Self> {
        match unsafe {
            self.lock
                // SAFETY: Temporary value; Not used to mutate anything.
                .clone()
                // SAFETY: `ArcMappedRwLock` does not provide mutable
                //         access to this lock so no write operations can be performed using it.
                .split_at_checked(mid)
        } {
            Ok((left, right)) => {
                self.lock = left;
                let (metadata, allocator) = (self.metadata, self.allocator.clone());
                Ok((
                    self,
                    Self {
                        lock: right,
                        metadata,
                        allocator,
                    },
                ))
            }
            Err(_) => Err(self),
        }
    }

    pub fn split_first(mut self) -> Result<(ArcMappedRwLock<T, U, A>, Self), Self> {
        match self.split_off_first() {
            Some(first) => Ok((first, self)),
            None => Err(self),
        }
    }

    pub fn split_last(mut self) -> Result<(ArcMappedRwLock<T, U, A>, Self), Self> {
        match self.split_off_last() {
            Some(last) => Ok((last, self)),
            None => Err(self),
        }
    }

    pub fn split_off<R: OneSidedRange<usize>>(&mut self, range: R) -> Option<Self> {
        // SAFETY: `ArcMappedRwLock` does not provide mutable
        //         access to this lock so no write operations can be performe
        match unsafe { self.lock.split_off(range) } {
            Some(lock) => Some(Self {
                lock,
                metadata: self.metadata,
                allocator: self.allocator.clone(),
            }),
            None => None,
        }
    }

    pub fn split_off_first(&mut self) -> Option<ArcMappedRwLock<T, U, A>> {
        // SAFETY: `ArcMappedRwLock` does not provide mutable
        //         access to this lock so no write operations can be performe
        match unsafe { self.lock.split_off_first() } {
            Some(lock) => Some(ArcMappedRwLock {
                lock,
                metadata: self.metadata,
                allocator: self.allocator.clone(),
            }),
            None => None,
        }
    }

    pub fn split_off_last(&mut self) -> Option<ArcMappedRwLock<T, U, A>> {
        // SAFETY: `ArcMappedRwLock` does not provide mutable
        //         access to this lock so no write operations can be performe
        match unsafe { self.lock.split_off_last() } {
            Some(lock) => Some(ArcMappedRwLock {
                lock,
                metadata: self.metadata,
                allocator: self.allocator.clone(),
            }),
            None => None,
        }
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> Drop for ArcMappedRwLock<T, U, A> {
    fn drop(&mut self) {
        // SAFETY: - By construction, the pointee of `self.lock` has been allocated as a part of an `InnerArc`.
        //         - By construction, `self.metadata` is the metadata of the underlying allocation.
        let (allocation, layout) =
            unsafe { InnerArc::<U>::from_inner_poison_lock_with_metadata(self.lock.get_lock(), self.metadata) };
        if unsafe { InnerArc::decrement_shared_counter(allocation, Ordering::Release) } {
            atomic::fence(Ordering::Acquire);
            if const { mem::needs_drop::<InnerArc<U>>() } {
                // SAFETY: - By construction, `allocation` points to live and valid data.
                //         - Ensured this was the last handle to this allocation.
                unsafe {
                    allocation.drop_in_place();
                }
            }
            // SAFETY: By construction, this allocation has been allocated by this allocator.
            unsafe {
                self.allocator.deallocate(allocation.cast(), layout);
            }
        }
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> Deref for ArcMappedRwLock<T, U, A> {
    type Target = MappedRwLock<T>;

    fn deref(&self) -> &MappedRwLock<T> {
        &self.lock
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> AsRef<MappedRwLock<T>> for ArcMappedRwLock<T, U, A> {
    fn as_ref(&self) -> &MappedRwLock<T> {
        &self.lock
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> Borrow<MappedRwLock<T>> for ArcMappedRwLock<T, U, A> {
    fn borrow(&self) -> &MappedRwLock<T> {
        &self.lock
    }
}

impl<T, U, V, A> CoerceUnsized<ArcMappedRwLock<V, U, A>> for ArcMappedRwLock<T, U, A>
where
    T: Unsize<V> + ?Sized,
    U: ?Sized,
    A: Allocator,
{
}

unsafe impl<T, U, A> Send for ArcMappedRwLock<T, U, A>
where
    T: Send + Sync + ?Sized,
    U: Send + Sync + ?Sized,
    A: Allocator + Send,
{
}

unsafe impl<T, U, A> Sync for ArcMappedRwLock<T, U, A>
where
    T: Send + Sync + ?Sized,
    U: Send + Sync + ?Sized,
    A: Allocator + Sync,
{
}

impl<T: ?Sized, U: ?Sized, A: AllocatorClone> Clone for ArcMappedRwLock<T, U, A> {
    fn clone(&self) -> Self {
        if unsafe {
            // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
            InnerArc::increment_shared_counter(
                // SAFETY: - By construction, the pointee of `self.lock` has been allocated as a part of an `InnerArc`.
                //         - By construction, `self.metadata` is the metadata of the underlying allocation.
                InnerArc::<U>::from_inner_poison_lock_with_metadata(self.lock.get_lock(), self.metadata).0,
                Ordering::Release,
            )
        } {
            process::abort();
        }
        Self {
            // SAFETY: `ArcMappedRwLock` does not provide mutable
            //         access to this lock so no write operations can be performed using it.
            lock: unsafe { self.lock.clone() },
            metadata: self.metadata,
            allocator: self.allocator.clone(),
        }
    }
}
