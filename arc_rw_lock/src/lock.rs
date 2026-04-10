mod inner;
pub(crate) use inner::InnerRwLock;

mod mapped {
    use crate::lock::InnerRwLock;

    use super::inner::PoisonLock;
    use std::{
        marker::PhantomData,
        ops::{Deref, DerefMut},
        ptr::NonNull,
        sync::nonpoison::WouldBlock,
        thread::panicking,
    };

    pub struct MappedRwLock<T: ?Sized, U: ?Sized = dyn Send + Sync + 'static> {
        pub(crate) inner: NonNull<InnerRwLock<U>>,
        pub(crate) subfield: NonNull<T>,
    }

    impl<T: ?Sized, U: ?Sized> MappedRwLock<T, U> {
        pub fn read(&self) -> &T {
            // SAFETY: - By construcion, `self.subfield` points to live and valid data.
            //         - By construcion, no other lock has mutable access to
            //           the subfield.
            unsafe { self.subfield.as_ref() }
        }

        pub fn write(&mut self) -> MappedRwLockGuard<'_, T> {
            // SAFETY: By construction, `self.inner` points to live and valid data.
            let poison_lock = unsafe { &(*self.inner.as_ptr()).poison_lock };
            poison_lock.lock.write();
            MappedRwLockGuard {
                lock: poison_lock,
                // SAFETY: - By construction, `self.subfield` points to live and valid data.
                //         - Aliasing rules are enforced via synchronization.
                data: unsafe { self.subfield.as_mut() },
                phantom: PhantomData,
            }
        }

        pub fn try_write(&mut self) -> Result<MappedRwLockGuard<'_, T>, WouldBlock> {
            // SAFETY: By construction, `self.inner` points to live and valid data.
            let poison_lock = unsafe { &(*self.inner.as_ptr()).poison_lock };
            if poison_lock.lock.try_write() {
                Ok(MappedRwLockGuard {
                    lock: poison_lock,
                    // SAFETY: - By construction, `self.subfield` points to live and valid data.
                    //         - Aliasing rules are enforced via synchronization.
                    data: unsafe { self.subfield.as_mut() },
                    phantom: PhantomData,
                })
            } else {
                Err(WouldBlock)
            }
        }
    }

    unsafe impl<T: Send + Sync + ?Sized> Sync for MappedRwLock<T> {}

    pub struct MappedRwLockGuard<'a, T: ?Sized> {
        lock: &'a PoisonLock,
        data: &'a mut T,
        /// For opting-out of `Send`
        phantom: PhantomData<*const T>,
    }

    impl<'a, T: ?Sized> Drop for MappedRwLockGuard<'a, T> {
        fn drop(&mut self) {
            // SAFETY: The existance of this guard guarantees that the counter is non-zero.
            unsafe {
                self.lock.lock.drop_writer_unchecked();
            }
            if panicking() {
                self.lock.poison();
            }
        }
    }

    impl<'a, T: ?Sized> Deref for MappedRwLockGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            self.data
        }
    }

    impl<'a, T: ?Sized> DerefMut for MappedRwLockGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.data
        }
    }

    unsafe impl<'a, T: Sync + ?Sized> Sync for MappedRwLockGuard<'a, T> {}
}
pub use mapped::{MappedRwLock, MappedRwLockGuard};

mod read {
    use super::inner::InnerRwLock;
    use std::{
        marker::PhantomData,
        ops::Deref,
        ptr::NonNull,
        sync::{LockResult, PoisonError, TryLockError, TryLockResult},
    };

    pub struct ReaderLock<T: ?Sized>(pub(crate) NonNull<InnerRwLock<T>>);

    impl<T: ?Sized> ReaderLock<T> {
        pub fn read(&self) -> LockResult<ReaderLockGuard<'_, T>> {
            // SAFETY: By construction, `self.0` points to live and valid data.
            let lock = unsafe { &(*self.0.as_ptr()).poison_lock };
            lock.lock.read_whole();
            let guard = ReaderLockGuard {
                lock: self.0,
                phantom: PhantomData,
            };
            if lock.is_poisoned() {
                Ok(guard)
            } else {
                Err(PoisonError::new(guard))
            }
        }

        pub fn try_read(&self) -> TryLockResult<ReaderLockGuard<'_, T>> {
            // SAFETY: By construction, `self.0` points to live and valid data.
            let poison_lock = unsafe { &(*self.0.as_ptr()).poison_lock };
            if poison_lock.lock.try_read_whole() {
                let guard = ReaderLockGuard {
                    lock: self.0,
                    phantom: PhantomData,
                };
                if poison_lock.is_poisoned() {
                    Err(TryLockError::Poisoned(PoisonError::new(guard)))
                } else {
                    Ok(guard)
                }
            } else {
                Err(TryLockError::WouldBlock)
            }
        }
    }

    unsafe impl<T: Send + Sync + ?Sized> Send for ReaderLock<T> {}

    unsafe impl<T: Send + Sync + ?Sized> Sync for ReaderLock<T> {}

    pub struct ReaderLockGuard<'a, T: ?Sized> {
        lock: NonNull<InnerRwLock<T>>,
        phantom: PhantomData<&'a T>,
    }

    impl<'a, T: ?Sized> Drop for ReaderLockGuard<'a, T> {
        fn drop(&mut self) {
            unsafe {
                // SAFETY: By construction, `self.lock` points to live and valid data.
                (*self.lock.as_ptr())
                    .poison_lock
                    .lock
                    // SAFETY: The existance of this guard guarantees that the counter is non-zero.
                    .drop_whole_reader_unchecked();
            }
        }
    }

    impl<'a, T: ?Sized> Deref for ReaderLockGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            // SAFETY: - By construction, `self.lock` points to live and valid data.
            //         - Aliasing rules are enforced via synchronization.
            unsafe { &(*self.lock.as_ptr()).data }
        }
    }

    unsafe impl<'a, T: Sync + ?Sized> Sync for ReaderLockGuard<'a, T> {}
}
pub use read::{ReaderLock, ReaderLockGuard};
