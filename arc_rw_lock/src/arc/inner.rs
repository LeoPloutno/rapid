use std::{
    alloc::Layout,
    hint,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::lock::InnerRwLock;

#[repr(C)]
pub(crate) struct InnerArc<T: ?Sized> {
    counter: AtomicUsize,
    lock: InnerRwLock<T>,
}

impl<T: ?Sized> InnerArc<T> {
    const SHARED_COUNTER_ONE: usize = 1;
    const UNIQUE_COUNTER_ONE: usize = 1 << (usize::BITS / 2);
    const SHARED_COUNTER_MAX: usize = {
        let mut accum = 0;
        let mut i = 0;
        while i < usize::BITS / 2 {
            accum = (accum << 1) | 1;
            i += 1;
        }
        accum
    };
    const UNIQUE_COUNTER_MAX: usize = Self::SHARED_COUNTER_MAX << (usize::BITS / 2);

    pub(crate) const unsafe fn from_lock(lock: NonNull<InnerRwLock<T>>) -> (NonNull<Self>, Layout) {
        let (layout, offset) = match Layout::new::<AtomicUsize>()
            // SAFETY: User-upheld invariant.
            .extend(unsafe { Layout::for_value_raw(lock.as_ptr()) })
        {
            Ok(res) => res,
            // SAFETY: User-upheld invariant.
            Err(_) => unsafe { hint::unreachable_unchecked() },
        };
        let (ptr, metadata) = lock.to_raw_parts();
        // SAFETY: By construction, `ptr.byte_sub(offset)` calculates the
        //         address of the underlying `InnerArc`, which has already
        //         been successfully allocated.
        (
            NonNull::from_raw_parts(unsafe { ptr.byte_sub(offset) }, metadata),
            layout,
        )
    }

    pub(crate) unsafe fn decrement_shared_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_sub(Self::SHARED_COUNTER_ONE, order) == Self::SHARED_COUNTER_ONE
    }

    pub(crate) unsafe fn decrement_unique_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_sub(Self::UNIQUE_COUNTER_ONE, order) == Self::UNIQUE_COUNTER_ONE
    }

    pub(crate) unsafe fn increment_shared_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_add(Self::SHARED_COUNTER_ONE, order) == Self::SHARED_COUNTER_MAX
    }

    pub(crate) unsafe fn increment_unique_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_add(Self::UNIQUE_COUNTER_ONE, order) == Self::UNIQUE_COUNTER_MAX
    }
}
