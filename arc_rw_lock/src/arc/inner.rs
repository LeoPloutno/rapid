use std::{
    alloc::{Layout, LayoutError},
    hint,
    ptr::{NonNull, Pointee},
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::lock::{InnerRwLock, PoisonLock};

#[repr(C)]
pub(crate) struct InnerArc<T: ?Sized> {
    pub(crate) counter: AtomicUsize,
    pub(crate) lock: InnerRwLock<T>,
}

impl<T: ?Sized> InnerArc<T> {
    pub(crate) const SHARED_COUNTER_ONE: usize = 1;
    pub(crate) const UNIQUE_COUNTER_ONE: usize = 1 << (usize::BITS / 2);
    pub(crate) const SHARED_COUNTER_MAX: usize = {
        let mut accum = 0;
        let mut i = 0;
        while i < usize::BITS / 2 {
            accum = (accum << 1) | 1;
            i += 1;
        }
        accum
    };
    pub(crate) const UNIQUE_COUNTER_MAX: usize = Self::SHARED_COUNTER_MAX << (usize::BITS / 2);

    pub(crate) const fn get_layout(data_layout: Layout) -> Result<Layout, LayoutError> {
        match Layout::new::<AtomicUsize>().extend(match InnerRwLock::<T>::get_layout(data_layout) {
            Ok(layout) => layout,
            err => return err,
        }) {
            Ok((layout, _)) => Ok(layout.pad_to_align()),
            Err(err) => return Err(err),
        }
    }

    /// Converts a type-erased pointer to a `PoisonLock` into a pointer to an `InnerArc<T>`.
    ///
    /// # Safety
    ///
    /// - The pointer points to a valid and live `PoisonLock`.
    /// - The pointer was derived from a pointer to a valid and live `InnerArc<T>`
    ///   with the same metadata as `metadata`
    #[inline]
    pub(crate) const unsafe fn from_poison_lock_with_metadata(
        lock: NonNull<PoisonLock>,
        metadata: <T as Pointee>::Metadata,
    ) -> (NonNull<Self>, Layout) {
        // SAFETY: User-upheld invariants.
        unsafe { Self::from_inner_rw_lock(InnerRwLock::from_poison_lock_with_metadata(lock, metadata)) }
    }

    /// Converts a pointer to a `InnerRwLock<T>` into a pointer to an `InnerArc<T>`.
    ///
    /// # Safety
    ///
    /// - The pointer points to a valid and live `InnerRwLock<T>`.
    /// - The pointer was derived from a pointer to a valid and live `InnerArc<T>`.
    #[inline]
    pub(crate) const unsafe fn from_inner_rw_lock(lock: NonNull<InnerRwLock<T>>) -> (NonNull<Self>, Layout) {
        let (layout, offset) = match Layout::new::<AtomicUsize>()
            // SAFETY: User-upheld invariant.
            .extend(unsafe { Layout::for_value_raw(lock.as_ptr()) })
        {
            Ok(res) => res,
            // SAFETY: User-upheld invariant.
            Err(_) => unsafe { hint::unreachable_unchecked() },
        };
        let (ptr, metadata) = lock.to_raw_parts();
        (
            // SAFETY: By construction, `ptr.byte_sub(offset)` calculates the
            //         address of the underlying `InnerArc`, which has already
            //         been successfully allocated.
            NonNull::from_raw_parts(unsafe { ptr.byte_sub(offset) }, metadata),
            layout,
        )
    }

    /// Decrements the shared counter assuming it is non-zero and returns
    /// whether it was unity.
    ///
    /// # Safety
    ///
    /// - The pointer points to a valid and live `InnerArc<T>`.
    /// - The shared counter is non-zero.
    pub(crate) unsafe fn decrement_shared_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_sub(Self::SHARED_COUNTER_ONE, order) == Self::SHARED_COUNTER_ONE
    }

    /// Decrements the unique counter assuming it is non-zero and returns
    /// whether it was unity.
    ///
    /// # Safety
    ///
    /// - The pointer points to a valid and live `InnerArc<T>`.
    /// - The unique counter is non-zero.
    pub(crate) unsafe fn decrement_unique_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_sub(Self::UNIQUE_COUNTER_ONE, order) == Self::UNIQUE_COUNTER_ONE
    }

    /// Increments the shared counter and returns whether the operation overflowed.
    ///
    /// # Safety
    ///
    /// The pointer points to a valid and live `InnerArc<T>`.
    pub(crate) unsafe fn increment_shared_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_add(Self::SHARED_COUNTER_ONE, order) == Self::SHARED_COUNTER_MAX
    }

    /// Increments the unique counter and returns whether the operation overflowed.
    ///
    /// # Safety
    ///
    /// The pointer points to a valid and live `InnerArc<T>`.
    pub(crate) unsafe fn increment_unique_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_add(Self::UNIQUE_COUNTER_ONE, order) == Self::UNIQUE_COUNTER_MAX
    }
}
