use std::{
    alloc::{Allocator, Global, Layout},
    hint,
    mem::needs_drop,
    ptr::NonNull,
    sync::atomic::{self, AtomicUsize, Ordering},
};

use crate::{inner::InnerRwLock, lock::MappedRwLock};

#[repr(C)]
pub(crate) struct InnerArc<T: ?Sized> {
    counter: AtomicUsize,
    lock: InnerRwLock<T>,
}

impl<T: ?Sized> InnerArc<T> {
    const SHARED_COUNTER_ONE: usize = 1;
    const UNIQUE_COUNTER_ONE: usize = 1 << usize::BITS / 2;
    const SHARED_COUNTER_MAX: usize = {
        let mut accum = 0;
        let mut i = 0;
        while i < usize::BITS / 2 {
            accum = (accum << 1) | 1;
            i += 1;
        }
        accum
    };
    const UNIQUE_COUNTER_MAX: usize = Self::SHARED_COUNTER_MAX << usize::BITS / 2;

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

pub struct ArcMappedRwLock<T: ?Sized, U: ?Sized = dyn 'static + Send + Sync, A: Allocator = Global> {
    pub(crate) lock: MappedRwLock<T, U>,
    pub(crate) allocator: A,
}

impl<T: ?Sized, U: ?Sized, A: Allocator> Drop for ArcMappedRwLock<T, U, A> {
    fn drop(&mut self) {
        // SAFETY: `self.lock.inner` has been allocated as a part of an `InnerArc`.
        let (allocation, layout) = unsafe { InnerArc::from_lock(self.lock.inner) };
        if unsafe { InnerArc::decrement_shared_counter(allocation, Ordering::Release) } {
            atomic::fence(Ordering::Acquire);
            if const { needs_drop::<InnerArc<U>>() } {
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

impl<T: ?Sized, U: ?Sized, A: Allocator> AsRef<MappedRwLock<T, U>> for ArcMappedRwLock<T, U, A> {
    fn as_ref(&self) -> &MappedRwLock<T, U> {
        &self.lock
    }
}

pub struct UniqueArcMappedRwLock<T: ?Sized, U: ?Sized = dyn 'static + Send + Sync, A: Allocator = Global> {
    pub(crate) lock: MappedRwLock<T, U>,
    pub(crate) allocator: A,
}

impl<T: ?Sized, U: ?Sized, A: Allocator> Drop for UniqueArcMappedRwLock<T, U, A> {
    fn drop(&mut self) {
        // SAFETY: `self.lock.inner` has been allocated as a part of an `InnerArc`.
        let (allocation, layout) = unsafe { InnerArc::from_lock(self.lock.inner) };
        if unsafe { InnerArc::decrement_unique_counter(allocation, Ordering::Release) } {
            atomic::fence(Ordering::Acquire);
            if const { needs_drop::<InnerArc<U>>() } {
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

impl<T: ?Sized, U: ?Sized, A: Allocator> AsRef<MappedRwLock<T, U>> for UniqueArcMappedRwLock<T, U, A> {
    fn as_ref(&self) -> &MappedRwLock<T, U> {
        &self.lock
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> AsMut<MappedRwLock<T, U>> for UniqueArcMappedRwLock<T, U, A> {
    fn as_mut(&mut self) -> &mut MappedRwLock<T, U> {
        &mut self.lock
    }
}
