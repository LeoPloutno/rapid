use crate::{ArcMappedRwLock, MappedRwLock, arc::InnerArc};
use std::{
    alloc::{Allocator, AllocatorClone, Global},
    iter::FusedIterator,
    mem, process,
    ptr::Pointee,
    sync::atomic::{self, Ordering},
};

pub struct Iter<T, U: ?Sized = dyn Send + Sync + 'static, A: Allocator = Global> {
    lock: MappedRwLock<[T]>,
    metadata: <U as Pointee>::Metadata,
    allocator: A,
}

impl<T, U: ?Sized, A: Allocator> Iter<T, U, A> {
    /// Creates a new instance of `Iter`.
    ///
    /// # Safety
    ///
    /// See [`ArcMappedRwLock::from_lock`].
    #[inline]
    pub(crate) const unsafe fn from_lock(
        lock: MappedRwLock<[T]>,
        metadata: <U as Pointee>::Metadata,
        allocator: A,
    ) -> Self {
        Self {
            lock,
            metadata,
            allocator,
        }
    }

    /// Increments the unique counter.
    ///
    /// # Safety
    ///
    /// Must be used with a corresponding construction of a shared lock
    /// so that the incremented counter is accurate.
    unsafe fn increment_counter(&self) {
        if crate::unlikely(unsafe {
            // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
            InnerArc::increment_shared_counter(
                // SAFETY: - By construction, the pointee of `self.lock` has been allocated as a part of an `InnerArc`.
                //         - By construction, `self.metadata` is the metadata of the underlying allocation.
                InnerArc::<U>::from_poison_lock_with_metadata(self.lock.get_lock(), self.metadata).0,
                Ordering::Relaxed,
            )
        }) {
            process::abort();
        }
    }
}

impl<T, U: ?Sized, A: Allocator> Drop for Iter<T, U, A> {
    fn drop(&mut self) {
        // SAFETY: - By construction, the pointee of `self.lock` has been allocated as a part of an `InnerArc`.
        //         - By construction, `self.metadata` is the metadata of the underlying allocation.
        let (allocation, layout) =
            unsafe { InnerArc::<U>::from_poison_lock_with_metadata(self.lock.get_lock(), self.metadata) };

        // SAFETY: THe existence of `self` proves that the counter is non-zero.
        if unsafe { InnerArc::decrement_shared_counter_unchecked(allocation, Ordering::Release) } {
            return;
        }

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

unsafe impl<T, U, A> Send for Iter<T, U, A>
where
    T: Send + Sync,
    U: Send + Sync + ?Sized,
    A: Allocator + Send,
{
}

unsafe impl<T, U, A> Sync for Iter<T, U, A>
where
    T: Send + Sync,
    U: Send + Sync + ?Sized,
    A: Allocator + Sync,
{
}

impl<T, U: ?Sized, A: AllocatorClone> Iterator for Iter<T, U, A> {
    type Item = ArcMappedRwLock<T, U, A>;

    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: The constructed lock does not provide mutable access to the data.
        match unsafe { self.lock.split_off_first() } {
            Some(lock) => {
                // SAFETY: Below we construct another shared lock.
                unsafe {
                    self.increment_counter();
                }

                // SAFETY: By construction, this allocation has been allocated by this allocator.
                unsafe { Some(ArcMappedRwLock::from_lock(lock, self.metadata, self.allocator.clone())) }
            }
            None => None,
        }
    }
}

impl<T, U: ?Sized, A: AllocatorClone> DoubleEndedIterator for Iter<T, U, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // SAFETY: The constructed lock does not provide mutable access to the data.
        match unsafe { self.lock.split_off_last() } {
            Some(lock) => {
                // SAFETY: Below we construct another shared lock.
                unsafe {
                    self.increment_counter();
                }

                // SAFETY: By construction, this allocation has been allocated by this allocator.
                unsafe { Some(ArcMappedRwLock::from_lock(lock, self.metadata, self.allocator.clone())) }
            }
            None => None,
        }
    }
}

impl<T, U: ?Sized, A: AllocatorClone> ExactSizeIterator for Iter<T, U, A> {
    fn len(&self) -> usize {
        self.lock.get_data().len()
    }
}

impl<T, U: ?Sized, A: AllocatorClone> FusedIterator for Iter<T, U, A> {}

impl<T, U: ?Sized, A: AllocatorClone> Clone for Iter<T, U, A> {
    fn clone(&self) -> Self {
        // SAFETY: Below we construct another shared lock.
        unsafe {
            self.increment_counter();
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
