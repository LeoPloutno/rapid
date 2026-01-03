use std::{
    alloc::{Allocator, Global, Layout},
    mem::needs_drop,
    ptr::{NonNull, metadata},
    sync::atomic::{self, AtomicUsize, Ordering},
};

use crate::{inner::InnerRwLock, lock::MappedRwLock};

#[repr(C)]
struct InnerArc<T: ?Sized> {
    counter: AtomicUsize,
    lock: InnerRwLock<T>,
}

impl<T: ?Sized> InnerArc<T> {
    const SHARED_COUNTER_ONE: usize = 1;
    const UNIQUE_COUNTER_ONE: usize = 1 << usize::BITS / 2;
    const COUNTER_MAX: usize = {
        let mut accum = 0;
        let mut i = 0;
        while i < usize::BITS / 2 {
            accum = (accum << 1) | 1;
            i += 1;
        }
        accum
    };

    unsafe fn decrement_shared_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_sub(Self::SHARED_COUNTER_ONE, order)
            == Self::SHARED_COUNTER_ONE
    }

    unsafe fn decrement_unique_counter(this: NonNull<Self>, order: Ordering) -> bool {
        unsafe { &(*this.as_ptr()).counter }.fetch_sub(Self::UNIQUE_COUNTER_ONE, order)
            == Self::UNIQUE_COUNTER_ONE
    }
}

pub struct ArcMappedRwLock<T: ?Sized, U: ?Sized = dyn 'static + Send + Sync, A: Allocator = Global>
{
    pub(crate) lock: MappedRwLock<T, U>,
    pub(crate) allocator: A,
}

impl<T: ?Sized, U: ?Sized, A: Allocator> Drop for ArcMappedRwLock<T, U, A> {
    fn drop(&mut self) {
        let data_ptr = self.lock.inner;
        let (layout, offset) = Layout::new::<AtomicUsize>()
            // SAFETY: This layout has been calculated during allocation for this pointer.
            .extend(unsafe { Layout::for_value_raw(data_ptr.as_ptr()) })
            .unwrap();
        let (ptr, metadata) = data_ptr.to_raw_parts();
        // SAFETY: By construction, `ptr.byte_sub(offset)` calculates the
        //         address of the underlying `InnerArc`, which has already
        //         been successfully allocated.
        let allocation =
            NonNull::<InnerArc<U>>::from_raw_parts(unsafe { ptr.byte_sub(offset) }, metadata);
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

pub struct UniqueArcMappedRwLock<
    T: ?Sized,
    U: ?Sized = dyn 'static + Send + Sync,
    A: Allocator = Global,
> {
    pub(crate) lock: MappedRwLock<T, U>,
    pub(crate) allocator: A,
}

impl<T: ?Sized, U: ?Sized, A: Allocator> Drop for UniqueArcMappedRwLock<T, U, A> {
    fn drop(&mut self) {
        let data_ptr = self.lock.inner;
        let (layout, offset) = Layout::new::<AtomicUsize>()
            // SAFETY: This layout has been calculated during allocation for this pointer.
            .extend(unsafe { Layout::for_value_raw(data_ptr.as_ptr()) })
            .unwrap();
        let (ptr, metadata) = data_ptr.to_raw_parts();
        // SAFETY: By construction, `ptr.byte_sub(offset)` calculates the
        //         address of the underlying `InnerArc`, which has already
        //         been successfully allocated.
        let allocation =
            NonNull::<InnerArc<U>>::from_raw_parts(unsafe { ptr.byte_sub(offset) }, metadata);
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

impl<T: ?Sized, U: ?Sized, A: Allocator> AsRef<MappedRwLock<T, U>>
    for UniqueArcMappedRwLock<T, U, A>
{
    fn as_ref(&self) -> &MappedRwLock<T, U> {
        &self.lock
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> AsMut<MappedRwLock<T, U>>
    for UniqueArcMappedRwLock<T, U, A>
{
    fn as_mut(&mut self) -> &mut MappedRwLock<T, U> {
        &mut self.lock
    }
}
