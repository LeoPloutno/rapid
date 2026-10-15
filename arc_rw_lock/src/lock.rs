mod inner;
pub(crate) use inner::{InnerRwLock, PoisonLock};

mod mapped {
    use super::inner::PoisonLock;
    use std::{
        marker::{PhantomData, Unsize},
        mem::MaybeUninit,
        ops::{CoerceUnsized, Deref, DerefMut, OneSidedRange, OneSidedRangeBound},
        ptr::NonNull,
        sync::nonpoison::WouldBlock,
        thread::panicking,
    };

    pub struct MappedRwLock<T: ?Sized> {
        lock: NonNull<PoisonLock>,
        data: NonNull<T>,
    }

    impl<T: ?Sized> MappedRwLock<T> {
        /// Creates a new instance of `MappedRwLock` from raw parts.
        ///
        /// # Safety
        ///
        /// - `lock` points to a valid and live `PoisonLock`.
        /// - `data` points to a valid and live instance of `T`
        /// - Both `lock` and `data` are derived from a pointer to the same instance of `InnerArc`.
        /// - If there are other locks pointing to the pointee of `data`,
        ///   neither they nor the one returned by this function shall be used
        ///   to mutate the data.
        #[inline]
        pub(crate) const unsafe fn from_raw_parts(lock: NonNull<PoisonLock>, data: NonNull<T>) -> Self {
            Self { lock, data }
        }

        /// Clones this lock.
        ///
        /// # Safety
        ///
        /// If the lock is shared, neither this lock nor the locks returned by this function
        /// shall be used to mutate the data.
        #[inline]
        pub(crate) const unsafe fn clone(&self) -> Self {
            Self { ..*self }
        }

        #[inline]
        pub(crate) const fn get_data(&self) -> NonNull<T> {
            self.data
        }

        #[inline]
        pub(crate) const fn get_lock(&self) -> NonNull<PoisonLock> {
            self.lock
        }

        pub fn read(&self) -> &T {
            // SAFETY: - By construcion, `self.data` points to live and valid data.
            //         - By construcion, no other lock has mutable access to
            //           the data.
            unsafe { self.data.as_ref() }
        }

        pub fn write(&mut self) -> MappedRwLockGuard<'_, T> {
            // SAFETY: By construction, `self.lock` points to live and valid data.
            let poison_lock = unsafe { &(*self.lock.as_ptr()) };
            poison_lock.lock.write();
            MappedRwLockGuard {
                lock: poison_lock,
                // SAFETY: - By construction, `self.data` points to live and valid data.
                //         - Aliasing rules are enforced via synchronization.
                data: unsafe { self.data.as_mut() },
                phantom: PhantomData,
            }
        }

        pub fn try_write(&mut self) -> Result<MappedRwLockGuard<'_, T>, WouldBlock> {
            // SAFETY: By construction, `self.lock` points to live and valid data.
            let poison_lock = unsafe { &(*self.lock.as_ptr()) };
            if poison_lock.lock.try_write() {
                Ok(MappedRwLockGuard {
                    lock: poison_lock,
                    // SAFETY: - By construction, `self.data` points to live and valid data.
                    //         - Aliasing rules are enforced via synchronization.
                    data: unsafe { self.data.as_mut() },
                    phantom: PhantomData,
                })
            } else {
                Err(WouldBlock)
            }
        }
    }

    impl<T> MappedRwLock<[T]> {
        /// # Safety
        ///
        /// If the lock is shared, neither of the locks returned by this function
        /// shall be used to mutate the data.
        ///
        /// # Panics
        ///
        /// Panics if `mid > len`.
        pub(crate) const unsafe fn split_at(self, mid: usize) -> (Self, Self) {
            // SAFETY: User-upheld invariants.
            match unsafe { self.split_at_checked(mid) } {
                Ok(pair) => pair,
                Err(_) => panic!("mid > len"),
            }
        }

        /// # Safety
        ///
        /// See [`MappedRwLock::split_at`].
        #[inline]
        pub(crate) const unsafe fn split_at_checked(mut self, mid: usize) -> Result<(Self, Self), Self> {
            let (ptr, len) = self.data.to_raw_parts();
            if mid > len {
                return Err(self);
            }
            let ptr = ptr.cast::<T>();
            self.data = NonNull::from_raw_parts(ptr, mid);
            let lock = self.lock;
            Ok((self, unsafe {
                // SAFETY: - Re-used the same lock.
                //         - The computed pointer is derived from `self.data`, which itself is valid.
                Self::from_raw_parts(
                    lock,
                    NonNull::from_raw_parts(
                        // SAFETY: Checked above that `mid` is within the slice's range.
                        ptr.add(mid),
                        // SAFETY: Checker above that `mid <= len`, which implies `len - mid >= 0`.
                        len.unchecked_sub(mid),
                    ),
                )
            }))
        }

        /// # Safety
        ///
        /// See [`MappedRwLock::split_at`].
        #[inline]
        pub(crate) const unsafe fn split_first(mut self) -> Result<(MappedRwLock<T>, Self), Self> {
            // SAFETY: User-upheld invariants.
            match unsafe { self.split_off_first() } {
                Some(first) => Ok((first, self)),
                None => Err(self),
            }
        }

        /// # Safety
        ///
        /// See [`MappedRwLock::split_at`].
        #[inline]
        pub(crate) const unsafe fn split_last(mut self) -> Result<(MappedRwLock<T>, Self), Self> {
            // SAFETY: User-upheld invariants.
            match unsafe { self.split_off_last() } {
                Some(first) => Ok((first, self)),
                None => Err(self),
            }
        }

        /// # Safety
        ///
        /// If the lock is shared, neither it nor the lock returned by this function
        /// shall be used to mutate the data.
        #[inline]
        pub(crate) unsafe fn split_off<R: OneSidedRange<usize>>(&mut self, range: R) -> Option<Self> {
            let (bound_type, mut bound) = range.bound();
            let (ptr, len) = self.data.to_raw_parts();
            if bound > len {
                return None;
            }
            let ptr = ptr.cast::<T>();
            let data;
            (self.data, data) = match bound_type {
                OneSidedRangeBound::StartInclusive => {
                    (NonNull::from_raw_parts(ptr, bound), unsafe {
                        NonNull::from_raw_parts(
                            // SAFETY: Checked above that `bound` is within the slice's range.
                            ptr.add(bound),
                            // SAFETY: Checked above that `bound <= len`, which implies `len - bound >= 0`.
                            len.unchecked_sub(bound),
                        )
                    })
                }
                OneSidedRangeBound::End => {
                    (
                        unsafe {
                            NonNull::from_raw_parts(
                                // SAFETY: Checked above that `bound` is within the slice's range.
                                ptr.add(bound),
                                // SAFETY: Checked above that `bound <= len`, which implies `len - bound >= 0`.
                                len.unchecked_sub(bound),
                            )
                        },
                        NonNull::from_raw_parts(ptr, bound),
                    )
                }
                OneSidedRangeBound::EndInclusive => {
                    bound += 1;
                    if bound > len {
                        return None;
                    } else {
                        (
                            unsafe {
                                NonNull::from_raw_parts(
                                    // SAFETY: Checked above that `bound` is within the slice's range.
                                    ptr.add(bound),
                                    // SAFETY: Checked above that `bound <= len`, which implies `len - bound >= 0`.
                                    len.unchecked_sub(bound),
                                )
                            },
                            NonNull::from_raw_parts(ptr, bound),
                        )
                    }
                }
            };
            // SAFETY: - Re-used the same lock.
            //         - The computed pointer is derived from `self.data`, which itself is valid.
            Some(unsafe { Self::from_raw_parts(self.lock, data) })
        }

        /// # Safety
        ///
        /// See [`MappedRwLock::split_off`].
        #[inline]
        pub(crate) const unsafe fn split_off_first(&mut self) -> Option<MappedRwLock<T>> {
            let (ptr, len) = self.data.to_raw_parts();
            if len == 0 {
                return None;
            }
            let ptr = ptr.cast::<T>();
            self.data = unsafe {
                NonNull::from_raw_parts(
                    // SAFETY: Checked above that the slice contains at least one element.
                    ptr.add(1),
                    // SAFETY: Checked above that `len > 0`.
                    len.unchecked_sub(1),
                )
            };
            // SAFETY: - Re-used the same lock.
            //         - The computed pointer is derived from `self.data`, which itself is valid.
            Some(unsafe { MappedRwLock::from_raw_parts(self.lock, ptr) })
        }

        /// # Safety
        ///
        /// See [`MappedRwLock::split_off`].
        #[inline]
        pub(crate) const unsafe fn split_off_last(&mut self) -> Option<MappedRwLock<T>> {
            let (ptr, len) = self.data.to_raw_parts();
            if len == 0 {
                return None;
            }
            let ptr = ptr.cast::<T>();
            // SAFETY: Checked above that `len > 0`.
            let remaining_len = unsafe { len.unchecked_sub(1) };
            self.data = NonNull::from_raw_parts(ptr, remaining_len);
            Some(unsafe {
                // SAFETY: - Re-used the same lock.
                //         - The computed pointer is derived from `self.data`, which itself is valid.
                MappedRwLock::from_raw_parts(
                    self.lock,
                    // SAFETY: By construction, `remaining_len` is within the slice's range.
                    ptr.add(remaining_len),
                )
            })
        }
    }

    impl<T> MappedRwLock<MaybeUninit<T>> {
        #[inline]
        pub(crate) const unsafe fn assume_init(self) -> MappedRwLock<T> {
            MappedRwLock {
                lock: self.lock,
                data: self.data.cast(),
            }
        }
    }

    impl<T> MappedRwLock<[MaybeUninit<T>]> {
        #[inline]
        pub(crate) const unsafe fn assume_init(self) -> MappedRwLock<[T]> {
            let (ptr, len) = self.data.to_raw_parts();
            MappedRwLock {
                lock: self.lock,
                data: NonNull::from_raw_parts(ptr, len),
            }
        }
    }

    impl<T, V> CoerceUnsized<MappedRwLock<V>> for MappedRwLock<T>
    where
        T: Unsize<V> + ?Sized,
        V: ?Sized,
    {
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

    impl<'a, T, U> CoerceUnsized<MappedRwLockGuard<'a, U>> for MappedRwLockGuard<'a, T>
    where
        T: Unsize<U> + ?Sized,
        U: ?Sized,
    {
    }

    unsafe impl<'a, T: Sync + ?Sized> Sync for MappedRwLockGuard<'a, T> {}
}
pub use mapped::{MappedRwLock, MappedRwLockGuard};

mod read {
    use super::inner::InnerRwLock;
    use std::{
        marker::{PhantomData, Unsize},
        mem::MaybeUninit,
        ops::{CoerceUnsized, Deref},
        ptr::NonNull,
        sync::{LockResult, PoisonError, TryLockError, TryLockResult},
    };

    pub struct ReaderLock<T: ?Sized>(NonNull<InnerRwLock<T>>);

    impl<T: ?Sized> ReaderLock<T> {
        /// Creates a new instance of `ReaderLock` from raw parts.
        ///
        /// # Safety
        ///
        /// The pointer points to a valid and live `InnerRwLock`.
        #[inline]
        pub(crate) const unsafe fn from_raw_parts(lock: NonNull<InnerRwLock<T>>) -> Self {
            Self(lock)
        }

        #[inline]
        pub(crate) const fn get_lock(&self) -> NonNull<InnerRwLock<T>> {
            self.0
        }

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

    impl<T> ReaderLock<MaybeUninit<T>> {
        #[inline]
        pub(crate) const unsafe fn assume_init(self) -> ReaderLock<T> {
            ReaderLock(self.0.cast())
        }
    }

    impl<T> ReaderLock<[MaybeUninit<T>]> {
        #[inline]
        pub(crate) const unsafe fn assume_init(self) -> ReaderLock<[T]> {
            let (ptr, len) = self.0.to_raw_parts();
            ReaderLock(NonNull::from_raw_parts(ptr.cast::<T>(), len))
        }
    }

    impl<T, U> CoerceUnsized<ReaderLock<U>> for ReaderLock<T>
    where
        T: Unsize<U> + ?Sized,
        U: ?Sized,
    {
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

    impl<'a, T, U> CoerceUnsized<ReaderLockGuard<'a, U>> for ReaderLockGuard<'a, T>
    where
        T: Unsize<U> + ?Sized,
        U: ?Sized,
    {
    }

    unsafe impl<'a, T: Sync + ?Sized> Sync for ReaderLockGuard<'a, T> {}
}
pub use read::{ReaderLock, ReaderLockGuard};
