mod inner;
pub(crate) use inner::InnerArc;

mod mapped {
    use super::InnerArc;
    use crate::lock::MappedRwLock;
    use std::{
        alloc::{Allocator, Global},
        borrow::{Borrow, BorrowMut},
        convert::{AsMut, AsRef},
        mem::needs_drop,
        ops::{Deref, DerefMut},
        sync::atomic::{self, Ordering},
    };

    pub struct ArcMappedRwLock<T: ?Sized, U: ?Sized = dyn Send + Sync + 'static, A: Allocator = Global> {
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

    impl<T: ?Sized, U: ?Sized, A: Allocator> Deref for ArcMappedRwLock<T, U, A> {
        type Target = MappedRwLock<T, U>;

        fn deref(&self) -> &MappedRwLock<T, U> {
            &self.lock
        }
    }

    impl<T: ?Sized, U: ?Sized, A: Allocator> AsRef<MappedRwLock<T, U>> for ArcMappedRwLock<T, U, A> {
        fn as_ref(&self) -> &MappedRwLock<T, U> {
            &self.lock
        }
    }

    impl<T: ?Sized, U: ?Sized, A: Allocator> Borrow<MappedRwLock<T, U>> for ArcMappedRwLock<T, U, A> {
        fn borrow(&self) -> &MappedRwLock<T, U> {
            &self.lock
        }
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

    pub struct UniqueArcMappedRwLock<T: ?Sized, U: ?Sized = dyn Send + Sync + 'static, A: Allocator = Global> {
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

    impl<T: ?Sized, U: ?Sized, A: Allocator> Deref for UniqueArcMappedRwLock<T, U, A> {
        type Target = MappedRwLock<T, U>;

        fn deref(&self) -> &MappedRwLock<T, U> {
            &self.lock
        }
    }

    impl<T: ?Sized, U: ?Sized, A: Allocator> DerefMut for UniqueArcMappedRwLock<T, U, A> {
        fn deref_mut(&mut self) -> &mut MappedRwLock<T, U> {
            &mut self.lock
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

    impl<T: ?Sized, U: ?Sized, A: Allocator> Borrow<MappedRwLock<T, U>> for UniqueArcMappedRwLock<T, U, A> {
        fn borrow(&self) -> &MappedRwLock<T, U> {
            &self.lock
        }
    }

    impl<T: ?Sized, U: ?Sized, A: Allocator> BorrowMut<MappedRwLock<T, U>> for UniqueArcMappedRwLock<T, U, A> {
        fn borrow_mut(&mut self) -> &mut MappedRwLock<T, U> {
            &mut self.lock
        }
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
}
pub use mapped::{ArcMappedRwLock, UniqueArcMappedRwLock};

mod reader {
    use super::InnerArc;
    use crate::lock::ReaderLock;
    use std::{
        alloc::{Allocator, Global},
        borrow::Borrow,
        convert::AsRef,
        mem::needs_drop,
        ops::Deref,
        sync::atomic::{self, Ordering},
    };

    pub struct ArcReaderLock<T: ?Sized, A: Allocator = Global> {
        pub(crate) lock: ReaderLock<T>,
        pub(crate) allocator: A,
    }

    impl<T: ?Sized, A: Allocator> Drop for ArcReaderLock<T, A> {
        fn drop(&mut self) {
            // SAFETY: `self.lock.0` has been allocated as a part of an `InnerArc`.
            let (allocation, layout) = unsafe { InnerArc::from_lock(self.lock.0) };
            if unsafe { InnerArc::decrement_shared_counter(allocation, Ordering::Release) } {
                atomic::fence(Ordering::Acquire);
                if const { needs_drop::<InnerArc<T>>() } {
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

    impl<T: ?Sized, A: Allocator> Deref for ArcReaderLock<T, A> {
        type Target = ReaderLock<T>;

        fn deref(&self) -> &ReaderLock<T> {
            &self.lock
        }
    }

    impl<T: ?Sized, A: Allocator> AsRef<ReaderLock<T>> for ArcReaderLock<T, A> {
        fn as_ref(&self) -> &ReaderLock<T> {
            &self.lock
        }
    }

    impl<T: ?Sized, A: Allocator> Borrow<ReaderLock<T>> for ArcReaderLock<T, A> {
        fn borrow(&self) -> &ReaderLock<T> {
            &self.lock
        }
    }

    unsafe impl<T, A> Send for ArcReaderLock<T, A>
    where
        T: Send + Sync + ?Sized,
        A: Allocator + Send,
    {
    }

    unsafe impl<T, A> Sync for ArcReaderLock<T, A>
    where
        T: Send + Sync + ?Sized,
        A: Allocator + Sync,
    {
    }
}
pub use reader::ArcReaderLock;
