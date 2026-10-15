use crate::{UniqueArcMappedRwLock, arc::InnerArc, lock::InnerRwLock};
use std::{
    alloc::{Allocator, AllocatorClone, Global},
    iter::FusedIterator,
    mem,
    num::NonZeroUsize,
    process,
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};

pub struct ChunksExactMut<T, U: ?Sized = dyn Send + Sync + 'static, A: Allocator = Global> {
    remainder: NonNull<[T]>,
    slice: NonNull<[T]>,
    chunk_size: NonZeroUsize,
    allocation: NonNull<InnerRwLock<U>>,
    allocator: A,
}

impl<T, U: ?Sized, A: Allocator> ChunksExactMut<T, U, A> {
    /// Creates a new instance of `ChunksExact`.
    ///
    /// # Safety
    ///
    /// - `slice` points to a valid and live slice.
    /// - `allocation` points to a valid and live `InnerRwLock<U>`.
    /// - Both `slice` and `allocation` are derived from a pointer to the same instance of `InnerArc`.
    /// - There are no locks to the slice.
    #[inline]
    pub(crate) const unsafe fn from_raw_parts(
        slice: NonNull<[T]>,
        allocation: NonNull<InnerRwLock<U>>,
        chunk_size: NonZeroUsize,
        allocator: A,
    ) -> Self {
        let (ptr, len) = slice.to_raw_parts();
        let ptr = ptr.cast::<T>();
        let remainder = len % chunk_size.get();
        let len_chunks = len - remainder;
        Self {
            remainder: NonNull::from_raw_parts(
                // SAFETY: By construction, `ptr` points to a slice which contains at least `len_chunks + remainder` elements.
                unsafe { ptr.add(len_chunks) },
                remainder,
            ),
            slice: NonNull::from_raw_parts(ptr, len_chunks),
            chunk_size,
            allocation,
            allocator,
        }
    }

    pub const fn into_remainder(self) -> UniqueArcMappedRwLock<[T], U, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let data = unsafe { (&raw const self.remainder).read() };
        let allocation = unsafe { (&raw const self.allocation).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        unsafe {
            // SAFETY: - By construction, `data` and `allocation` are pointers derived
            //           from a pointer to a valid and live `InnerRwLock`.
            //         - Using the actual metadata of the allocation.
            //         - This lock is unique.
            UniqueArcMappedRwLock::from_raw_parts(
                // SAFETY: A pointer to a field of a live and valid struct is non-null.
                NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).poison_lock),
                data,
                allocation.to_raw_parts().1,
                allocator,
            )
        }
    }
}

impl<T, U: ?Sized, A: Allocator> Drop for ChunksExactMut<T, U, A> {
    fn drop(&mut self) {
        // SAFETY: `self.allocation` has been allocated as a part of an `InnerArc`.
        let (allocation, layout) = unsafe { InnerArc::from_inner_rw_lock(self.allocation) };
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

unsafe impl<T, U, A> Send for ChunksExactMut<T, U, A>
where
    T: Send + Sync,
    U: Send + Sync + ?Sized,
    A: Allocator + Send,
{
}

unsafe impl<T, U, A> Sync for ChunksExactMut<T, U, A>
where
    T: Send + Sync,
    U: Send + Sync + ?Sized,
    A: Allocator + Sync,
{
}

impl<T, U: ?Sized, A: AllocatorClone> Iterator for ChunksExactMut<T, U, A> {
    type Item = UniqueArcMappedRwLock<[T], U, A>;

    fn next(&mut self) -> Option<Self::Item> {
        let (ptr, len) = self.slice.to_raw_parts();
        if len > 0 {
            let ptr = ptr.cast::<T>();
            unsafe {
                self.slice = NonNull::from_raw_parts(
                    // SAFETY: By construction, 'ptr' points to a slice which contains at least `self.chunk_size` elements.
                    ptr.add(self.chunk_size.get()),
                    // SAFETY: Checked above that 'len > 0`.
                    //         By construction, `len` is a multiple of `self.chunk_size`.
                    len.unchecked_sub(self.chunk_size.get()),
                );
            }
            if crate::unlikely(unsafe {
                // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
                InnerArc::increment_unique_counter(
                    // SAFETY: `self.allocation` has been allocated as a part of an `InnerArc`.
                    InnerArc::from_inner_rw_lock(self.allocation).0,
                    Ordering::Release,
                )
            }) {
                process::abort()
            }
            Some(unsafe {
                // SAFETY: - By construction, the computed pointer and `self.allocation` are pointers derived
                //           from a pointer to a valid and live `InnerRwLock`.
                //         - Using the actual metadata of the allocation.
                //         - This lock is unique.
                UniqueArcMappedRwLock::from_raw_parts(
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    NonNull::new_unchecked(&raw mut (*self.allocation.as_ptr()).poison_lock),
                    NonNull::from_raw_parts(ptr, self.chunk_size.get()),
                    self.allocation.to_raw_parts().1,
                    self.allocator.clone(),
                )
            })
        } else {
            None
        }
    }
}

impl<T, U: ?Sized, A: AllocatorClone> DoubleEndedIterator for ChunksExactMut<T, U, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (ptr, mut len) = self.slice.to_raw_parts();
        if len > 0 {
            // SAFETY: Checked above that 'len > 0`.
            //         By construction, `len` is a multiple of `self.chunk_size`.
            len = unsafe { len.unchecked_sub(self.chunk_size.get()) };
            let ptr = ptr.cast::<T>();
            self.slice = NonNull::from_raw_parts(ptr, len);
            if crate::unlikely(unsafe {
                // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
                InnerArc::increment_unique_counter(
                    // SAFETY: `self.allocation` has been allocated as a part of an `InnerArc`.
                    InnerArc::from_inner_rw_lock(self.allocation).0,
                    Ordering::Release,
                )
            }) {
                process::abort()
            }
            Some(unsafe {
                // SAFETY: - By construction, the computed pointer and `self.allocation` are pointers derived
                //           from a pointer to a valid and live `InnerRwLock`.
                //         - Using the actual metadata of the allocation.
                //         - This lock is unique.
                UniqueArcMappedRwLock::from_raw_parts(
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    NonNull::new_unchecked(&raw mut (*self.allocation.as_ptr()).poison_lock),
                    NonNull::from_raw_parts(
                        // SAFETY: By construction, 'ptr' points to a slice which contains at least `len + self.chunk_size` elements.
                        ptr.add(len),
                        self.chunk_size.get(),
                    ),
                    self.allocation.to_raw_parts().1,
                    self.allocator.clone(),
                )
            })
        } else {
            None
        }
    }
}

impl<T, U: ?Sized, A: AllocatorClone> ExactSizeIterator for ChunksExactMut<T, U, A> {
    fn len(&self) -> usize {
        self.slice.len() / self.chunk_size
    }
}

impl<T, U: ?Sized, A: AllocatorClone> FusedIterator for ChunksExactMut<T, U, A> {}
