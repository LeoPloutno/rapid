use super::InnerArc;
use crate::{
    ArcMappedRwLock, ArcReaderLock, ReaderLock,
    lock::{InnerRwLock, MappedRwLock, PoisonLock},
    slice::{ChunksExactMut, IterMut},
};
use std::{
    alloc::{AllocError, Allocator, AllocatorClone, Global, Layout, LayoutError, handle_alloc_error},
    borrow::{Borrow, BorrowMut},
    convert::{AsMut, AsRef},
    marker::Unsize,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    ops::{CoerceUnsized, Deref, DerefMut, OneSidedRange},
    process,
    ptr::{NonNull, Pointee},
    range::Range,
    sync::atomic::{self, AtomicUsize, Ordering},
};

pub struct UniqueArcMappedRwLock<T: ?Sized, U: ?Sized = dyn Send + Sync + 'static, A: Allocator = Global> {
    pub(crate) lock: MappedRwLock<T>,
    pub(crate) metadata: <U as Pointee>::Metadata,
    pub(crate) allocator: A,
}

impl<T: ?Sized, U: ?Sized, A: Allocator> UniqueArcMappedRwLock<T, U, A> {
    /// Creates a new instance of `ArcMappedRwLock` from a `MapperRwLock`.
    ///
    /// # Safety
    ///
    /// - `metadata` is the same as the metadata of the
    ///   `InnerArc<U>` from which `lock` has been derived.
    /// - The underlying allocation has been allocated using `allocator` or a valid clone of its.
    pub(crate) const unsafe fn from_lock(
        lock: MappedRwLock<T>,
        metadata: <U as Pointee>::Metadata,
        allocator: A,
    ) -> Self {
        Self {
            lock,
            metadata,
            allocator,
        }
    }

    /// Creates a new instance of `UniqueArcMappedRwLock<T, U, A>` from raw parts.
    ///
    /// # Safety
    ///
    /// - `metadata` is the same as the metadata of the
    ///   `InnerArc<U>` from which `lock` and `data` are derived.
    /// - The underlying allocation has been allocated using `allocator` or a valid clone of its.
    /// - No other locks pointing to the pointee of `data` are accessed.
    /// - See [`MappedRwLock::from_raw_parts`]
    #[inline]
    pub(crate) const unsafe fn from_raw_parts(
        lock: NonNull<PoisonLock>,
        data: NonNull<T>,
        metadata: <U as Pointee>::Metadata,
        allocator: A,
    ) -> Self {
        Self {
            // SAFETY: User-upheld invariants.
            lock: unsafe { MappedRwLock::from_raw_parts(lock, data) },
            metadata,
            allocator,
        }
    }

    pub const fn allocator(this: &Self) -> &A {
        &this.allocator
    }

    pub fn into_arc(self) -> ArcMappedRwLock<T, U, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let metadata = unsafe { (&raw const self.metadata).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        unsafe {
            // SAFETY: - By construction, the pointee of `lock` has been allocated as a part of an `InnerArc`.
            //         - By construction, `metadata` is the metadata of the underlying allocation.
            let allocation = InnerArc::<U>::from_poison_lock_with_metadata(lock.get_lock(), metadata).0;
            // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
            if InnerArc::increment_shared_counter(allocation, Ordering::Relaxed) {
                process::abort();
            }
            InnerArc::decrement_unique_counter(allocation, Ordering::Release);
        };
        // SAFETY: By construction, all invariants are satisfied.
        unsafe { ArcMappedRwLock::from_lock(lock, metadata, allocator) }
    }
}

impl<T> UniqueArcMappedRwLock<T, T> {
    pub fn new(data: T) -> (Self, ArcReaderLock<T>) {
        Self::new_in(data, Global)
    }

    pub fn new_uninit() -> (
        UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
        ArcReaderLock<MaybeUninit<T>>,
    ) {
        Self::new_uninit_in(Global)
    }

    pub fn new_zeroed() -> (
        UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
        ArcReaderLock<MaybeUninit<T>>,
    ) {
        Self::new_zeroed_in(Global)
    }

    pub fn try_new(data: T) -> Result<(Self, ArcReaderLock<T>), AllocError> {
        Self::try_new_in(data, Global)
    }

    pub fn try_new_uninit() -> Result<
        (
            UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
            ArcReaderLock<MaybeUninit<T>>,
        ),
        AllocError,
    > {
        Self::try_new_uninit_in(Global)
    }

    pub fn try_new_zeroed() -> Result<
        (
            UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
            ArcReaderLock<MaybeUninit<T>>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_in(Global)
    }
}

impl<T, A: AllocatorClone> UniqueArcMappedRwLock<T, T, A> {
    #[inline]
    fn try_new_in_with_layout(data: T, alloc: A) -> Result<(Self, ArcReaderLock<T, A>), (Layout, AllocError)> {
        let inner = InnerArc {
            counter: AtomicUsize::new(InnerArc::<MaybeUninit<T>>::UNIQUE_COUNTER_ONE),
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
                // SAFETY: - Provided a clone of the allocator, where `A: AllocatorClone`.
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - This lock is unique.
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
            UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
            (&raw mut (*ptr).counter).write(AtomicUsize::new(InnerArc::<MaybeUninit<T>>::UNIQUE_COUNTER_ONE));
            (&raw mut (*ptr).lock.poison_lock).write(PoisonLock::new());
        }
        Ok((
            unsafe {
                // SAFETY: - Provided a clone of the allocator, where `A: AllocatorClone`.
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - This lock is unique.
                UniqueArcMappedRwLock::from_raw_parts(
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
            UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
            (&raw mut (*ptr).counter).write(AtomicUsize::new(InnerArc::<MaybeUninit<T>>::UNIQUE_COUNTER_ONE));
            (&raw mut (*ptr).lock.poison_lock).write(PoisonLock::new());
        }
        Ok((
            unsafe {
                // SAFETY: - Provided a clone of the allocator, where `A: AllocatorClone`.
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - This lock is unique.
                UniqueArcMappedRwLock::from_raw_parts(
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
        UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
        UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
            UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
            UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
            ArcReaderLock<MaybeUninit<T>, A>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_in_with_layout(alloc).map_err(|(_, err)| err)
    }
}

impl<T, A: Allocator> UniqueArcMappedRwLock<T, [T], A> {
    pub const fn element_offset(&self) -> usize {
        // SAFETY: - By construction, `self.lock` is a lock to valid and live data.
        //         - By construction, `self.metadata` is the metadata of the underlying allocation.
        let ptr_whole = unsafe {
            &raw mut (*InnerRwLock::<[T]>::from_poison_lock_with_metadata(self.lock.get_lock(), self.metadata).as_ptr())
                .data
        }
        .to_raw_parts()
        .0;
        // SAFETY: The offset of `data` guarantees `ptr` is non-null.
        let ptr_whole = unsafe { NonNull::new_unchecked(ptr_whole) }.cast::<T>();
        let ptr = self.lock.get_data();
        // SAFETY: By construction, `ptr` points to a subslice of `ptr_whole`.
        unsafe { ptr.offset_from_unsigned(ptr_whole) }
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

impl<T> UniqueArcMappedRwLock<[T], [T]> {
    pub fn new_uninit_slice(
        len: usize,
    ) -> (
        UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
        ArcReaderLock<[MaybeUninit<T>]>,
    ) {
        Self::new_uninit_slice_in(len, Global)
    }

    pub fn new_zeroed_slice(
        len: usize,
    ) -> (
        UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
        ArcReaderLock<[MaybeUninit<T>]>,
    ) {
        Self::new_zeroed_slice_in(len, Global)
    }

    pub fn try_new_uninit_slice(
        len: usize,
    ) -> Result<
        (
            UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
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
            UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
            ArcReaderLock<[MaybeUninit<T>]>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_slice_in(len, Global)
    }
}

impl<T, A: AllocatorClone> UniqueArcMappedRwLock<[T], [T], A> {
    #[inline]
    fn try_new_uninit_slice_in_with_layout(
        len: usize,
        alloc: A,
    ) -> Result<
        (
            UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
            unsafe {
                // SAFETY: - By construction, `len` is the length of the unsized part of the allocation.
                //         - Provided a clone of the allocator, where `A: AllocatorClone`.
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - This lock is unique.
                UniqueArcMappedRwLock::from_raw_parts(
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
            UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
            unsafe {
                // SAFETY: - By construction, `len` is the length of the unsized part of the allocation.
                //         - Provided a clone of the allocator, where `A: AllocatorClone`.
                //         - Both pointers are derived from a valid and live allocation
                //           of `InnerArc`.
                //         - This lock is unique.
                UniqueArcMappedRwLock::from_raw_parts(
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
        UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
        UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
            UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
            UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
            ArcReaderLock<[MaybeUninit<T>], A>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_slice_in_with_layout(len, alloc).map_err(|err| match err {
            TryAllocError::Layout(_) => AllocError,
            TryAllocError::Alloc { err, .. } => err,
        })
    }

    pub const fn subslice_range(&self) -> Range<usize> {
        // SAFETY: - By construction, `self.lock` is a lock to valid and live data.
        //         - By construction, `self.metadata` is the metadata of the underlying allocation.
        let ptr_whole = unsafe {
            &raw mut (*InnerRwLock::<[T]>::from_poison_lock_with_metadata(self.lock.get_lock(), self.metadata).as_ptr())
                .data
        }
        .to_raw_parts()
        .0;
        // SAFETY: The offset of `data` guarantees `ptr` is non-null.
        let ptr_whole = unsafe { NonNull::new_unchecked(ptr_whole) }.cast::<T>();
        let (ptr, len) = self.lock.get_data().to_raw_parts();
        let ptr = ptr.cast::<T>();
        // SAFETY: By construction, `ptr` points to a subslice of `ptr_whole`.
        let start = unsafe { ptr.offset_from_unsigned(ptr_whole) };
        Range {
            start,
            // SAFETY: By construction, `start + len` points within or right outside the allocation.
            end: unsafe { start.unchecked_add(len) },
        }
    }
}

impl<T, U: ?Sized, A: AllocatorClone> UniqueArcMappedRwLock<[T], U, A> {
    pub fn split_at_mut(self, mid: usize) -> (Self, Self) {
        match self.split_at_mut_checked(mid) {
            Ok(pair) => pair,
            Err(_) => panic!("mid > len"),
        }
    }

    pub fn split_at_mut_checked(mut self, mid: usize) -> Result<(Self, Self), Self> {
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

    pub fn split_first_mut(mut self) -> Result<(UniqueArcMappedRwLock<T, U, A>, Self), Self> {
        match self.split_off_first_mut() {
            Some(first) => Ok((first, self)),
            None => Err(self),
        }
    }

    pub fn split_last_mut(mut self) -> Result<(UniqueArcMappedRwLock<T, U, A>, Self), Self> {
        match self.split_off_last_mut() {
            Some(last) => Ok((last, self)),
            None => Err(self),
        }
    }

    pub fn split_off_mut<R: OneSidedRange<usize>>(&mut self, range: R) -> Option<Self> {
        // SAFETY: The resulting lock is unique.
        match unsafe { self.lock.split_off(range) } {
            Some(lock) => Some(Self {
                lock,
                metadata: self.metadata,
                allocator: self.allocator.clone(),
            }),
            None => None,
        }
    }

    pub fn split_off_first_mut(&mut self) -> Option<UniqueArcMappedRwLock<T, U, A>> {
        // SAFETY: The resulting lock is unique.
        match unsafe { self.lock.split_off_first() } {
            Some(lock) => Some(UniqueArcMappedRwLock {
                lock,
                metadata: self.metadata,
                allocator: self.allocator.clone(),
            }),
            None => None,
        }
    }

    pub fn split_off_last_mut(&mut self) -> Option<UniqueArcMappedRwLock<T, U, A>> {
        // SAFETY: The resulting lock is unique.
        match unsafe { self.lock.split_off_last() } {
            Some(lock) => Some(UniqueArcMappedRwLock {
                lock,
                metadata: self.metadata,
                allocator: self.allocator.clone(),
            }),
            None => None,
        }
    }
}

impl<T, U: ?Sized, A: Allocator> UniqueArcMappedRwLock<[T], U, A> {
    pub fn iter_mut(self) -> IterMut<T, U, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let metadata = unsafe { (&raw const self.metadata).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        // SAFETY: By construction, all invariants are satisfied.
        unsafe { IterMut::from_lock(lock, metadata, allocator) }
    }

    pub fn chunks_exact_mut(self, chunk_size: usize) -> ChunksExactMut<T, U, A> {
        assert!(chunk_size != 0, "chunk size must be non-zero");

        // SAFETY: Checked above that `chunk_size != 0`.
        let chunk_size = unsafe { NonZeroUsize::new_unchecked(chunk_size) };

        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let metadata = unsafe { (&raw const self.metadata).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);

        unsafe {
            // SAFETY: See below.
            ChunksExactMut::from_raw_parts(
                lock.get_data(),
                // SAFETY: - By construction, `lock` is a lock to valid and live data.
                //         - By construction, `metadata` is the metadata of the underlying allocation.
                InnerRwLock::from_poison_lock_with_metadata(lock.get_lock(), metadata),
                chunk_size,
                allocator,
            )
        }
    }
}

impl<T, U, A: Allocator> UniqueArcMappedRwLock<MaybeUninit<T>, MaybeUninit<U>, A> {
    pub const unsafe fn assume_init(self) -> UniqueArcMappedRwLock<T, U, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        UniqueArcMappedRwLock {
            // SAFETY: User-upheld invariant.
            lock: unsafe { lock.assume_init() },
            metadata: (),
            allocator,
        }
    }
}

impl<T, A: Allocator> UniqueArcMappedRwLock<MaybeUninit<T>, [MaybeUninit<T>], A> {
    pub const unsafe fn assume_init(self) -> UniqueArcMappedRwLock<T, [T], A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let metadata = unsafe { (&raw const self.metadata).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        UniqueArcMappedRwLock {
            // SAFETY: User-upheld invariant.
            lock: unsafe { lock.assume_init() },
            metadata,
            allocator,
        }
    }
}

impl<T, A: Allocator> UniqueArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A> {
    pub const unsafe fn assume_init(self) -> UniqueArcMappedRwLock<[T], [T], A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let metadata = unsafe { (&raw const self.metadata).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        UniqueArcMappedRwLock {
            // SAFETY: User-upheld invariant.
            lock: unsafe { lock.assume_init() },
            metadata,
            allocator,
        }
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> Drop for UniqueArcMappedRwLock<T, U, A> {
    fn drop(&mut self) {
        // SAFETY: - By construction, the pointee of `self.lock` has been allocated as a part of an `InnerArc`.
        //         - By construction, `self.metadata` is the metadata of the underlying allocation.
        let (allocation, layout) =
            unsafe { InnerArc::<U>::from_poison_lock_with_metadata(self.lock.get_lock(), self.metadata) };
        if unsafe { InnerArc::decrement_unique_counter(allocation, Ordering::Release) } {
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

impl<T: ?Sized, U: ?Sized, A: Allocator> Deref for UniqueArcMappedRwLock<T, U, A> {
    type Target = MappedRwLock<T>;

    fn deref(&self) -> &MappedRwLock<T> {
        &self.lock
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> DerefMut for UniqueArcMappedRwLock<T, U, A> {
    fn deref_mut(&mut self) -> &mut MappedRwLock<T> {
        &mut self.lock
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> AsRef<MappedRwLock<T>> for UniqueArcMappedRwLock<T, U, A> {
    fn as_ref(&self) -> &MappedRwLock<T> {
        &self.lock
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> AsMut<MappedRwLock<T>> for UniqueArcMappedRwLock<T, U, A> {
    fn as_mut(&mut self) -> &mut MappedRwLock<T> {
        &mut self.lock
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> Borrow<MappedRwLock<T>> for UniqueArcMappedRwLock<T, U, A> {
    fn borrow(&self) -> &MappedRwLock<T> {
        &self.lock
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> BorrowMut<MappedRwLock<T>> for UniqueArcMappedRwLock<T, U, A> {
    fn borrow_mut(&mut self) -> &mut MappedRwLock<T> {
        &mut self.lock
    }
}

impl<T, U, V, A> CoerceUnsized<UniqueArcMappedRwLock<V, U, A>> for UniqueArcMappedRwLock<T, U, A>
where
    T: Unsize<V> + ?Sized,
    U: ?Sized,
    A: Allocator,
{
}

unsafe impl<T, U, A> Send for UniqueArcMappedRwLock<T, U, A>
where
    T: Send + Sync + ?Sized,
    U: Send + Sync + ?Sized,
    A: Allocator + Send,
{
}

unsafe impl<T, U, A> Sync for UniqueArcMappedRwLock<T, U, A>
where
    T: Send + Sync + ?Sized,
    U: Send + Sync + ?Sized,
    A: Allocator + Sync,
{
}
