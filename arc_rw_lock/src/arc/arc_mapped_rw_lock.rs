use super::InnerArc;
use crate::{
    ArcReaderLock, ReaderLock,
    lock::{InnerRwLock, MappedRwLock, PoisonLock},
};
use std::{
    alloc::{AllocError, Allocator, AllocatorClone, Global, Layout, LayoutError, handle_alloc_error},
    borrow::Borrow,
    convert::AsRef,
    marker::Unsize,
    mem::{self, MaybeUninit, needs_drop},
    ops::{CoerceUnsized, Deref, DerefMut},
    process,
    ptr::NonNull,
    sync::atomic::{self, AtomicUsize, Ordering},
};

pub struct ArcMappedRwLock<T: ?Sized, U: ?Sized = dyn Send + Sync + 'static, A: Allocator = Global> {
    pub(crate) lock: MappedRwLock<T, U>,
    pub(crate) allocator: A,
}

impl<T: ?Sized, U: ?Sized, A: Allocator> ArcMappedRwLock<T, U, A> {
    pub const fn allocator(this: &Self) -> &A {
        &this.allocator
    }
}

impl<T> ArcMappedRwLock<T, T> {
    pub fn new(data: T) -> (Self, ArcReaderLock<T>) {
        Self::new_in(data, Global)
    }

    pub fn new_uninit() -> (
        ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
        ArcReaderLock<MaybeUninit<T>>,
    ) {
        Self::new_uninit_in(Global)
    }

    pub fn new_zeroed() -> (
        ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
        ArcReaderLock<MaybeUninit<T>>,
    ) {
        Self::new_zeroed_in(Global)
    }

    pub fn try_new(data: T) -> Result<(Self, ArcReaderLock<T>), AllocError> {
        Self::try_new_in(data, Global)
    }

    pub fn try_new_uninit() -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
            ArcReaderLock<MaybeUninit<T>>,
        ),
        AllocError,
    > {
        Self::try_new_uninit_in(Global)
    }

    pub fn try_new_zeroed() -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>>,
            ArcReaderLock<MaybeUninit<T>>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_in(Global)
    }
}

impl<T, A> ArcMappedRwLock<T, T, A>
where
    A: AllocatorClone,
{
    #[inline]
    fn try_new_in_with_layout(data: T, alloc: A) -> Result<(Self, ArcReaderLock<T, A>), (Layout, AllocError)> {
        let inner = InnerArc {
            counter: AtomicUsize::new(InnerArc::<MaybeUninit<T>>::SHARED_COUNTER_ONE),
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
        // SAFETY: A pointer to a field of a live and valid struct is non-null.
        let inner = unsafe { NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).lock) };
        Ok((
            Self {
                lock: MappedRwLock {
                    inner,
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    subfield: unsafe { NonNull::new_unchecked(&raw mut (*allocation.as_ptr()).lock.data) },
                },
                allocator: alloc.clone(),
            },
            ArcReaderLock {
                lock: ReaderLock(inner),
                allocator: alloc,
            },
        ))
    }

    #[inline]
    fn try_new_uninit_in_with_layout(
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
            (&raw mut (*ptr).counter).write(AtomicUsize::new(InnerArc::<MaybeUninit<T>>::SHARED_COUNTER_ONE));
            (&raw mut (*ptr).lock.poison_lock).write(PoisonLock::new());
        }
        // SAFETY: A pointer to a field of a live and valid struct is non-null.
        let inner = unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock) };
        Ok((
            ArcMappedRwLock {
                lock: MappedRwLock {
                    inner,
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    subfield: unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock.data) },
                },
                allocator: alloc.clone(),
            },
            ArcReaderLock {
                lock: ReaderLock(inner),
                allocator: alloc,
            },
        ))
    }

    #[inline]
    fn try_new_zeroed_in_with_layout(
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
            (&raw mut (*ptr).counter).write(AtomicUsize::new(InnerArc::<MaybeUninit<T>>::SHARED_COUNTER_ONE));
            (&raw mut (*ptr).lock.poison_lock).write(PoisonLock::new());
        }
        // SAFETY: A pointer to a field of a live and valid struct is non-null.
        let inner = unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock) };
        Ok((
            ArcMappedRwLock {
                lock: MappedRwLock {
                    inner,
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    subfield: unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock.data) },
                },
                allocator: alloc.clone(),
            },
            ArcReaderLock {
                lock: ReaderLock(inner),
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
        ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
        ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
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
            ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<T>, A>,
            ArcReaderLock<MaybeUninit<T>, A>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_in_with_layout(alloc).map_err(|(_, err)| err)
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

impl<T> ArcMappedRwLock<[T], [T]> {
    pub fn new_uninit_slice(
        len: usize,
    ) -> (
        ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
        ArcReaderLock<[MaybeUninit<T>]>,
    ) {
        Self::new_uninit_slice_in(len, Global)
    }

    pub fn new_zeroed_slice(
        len: usize,
    ) -> (
        ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
        ArcReaderLock<[MaybeUninit<T>]>,
    ) {
        Self::new_zeroed_slice_in(len, Global)
    }

    pub fn try_new_uninit_slice(
        len: usize,
    ) -> Result<
        (
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
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
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>]>,
            ArcReaderLock<[MaybeUninit<T>]>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_slice_in(len, Global)
    }
}

impl<T, A> ArcMappedRwLock<[T], [T], A>
where
    A: AllocatorClone,
{
    #[inline]
    fn try_new_uninit_slice_in_with_layout(
        len: usize,
        alloc: A,
    ) -> Result<
        (
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
        // SAFETY: A pointer to a field of a live and valid struct is non-null.
        let inner = unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock) };
        Ok((
            ArcMappedRwLock {
                lock: MappedRwLock {
                    inner,
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    subfield: unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock.data) },
                },
                allocator: alloc.clone(),
            },
            ArcReaderLock {
                lock: ReaderLock(inner),
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
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
        // SAFETY: A pointer to a field of a live and valid struct is non-null.
        let inner = unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock) };
        Ok((
            ArcMappedRwLock {
                lock: MappedRwLock {
                    inner,
                    // SAFETY: A pointer to a field of a live and valid struct is non-null.
                    subfield: unsafe { NonNull::new_unchecked(&raw mut (*ptr).lock.data) },
                },
                allocator: alloc.clone(),
            },
            ArcReaderLock {
                lock: ReaderLock(inner),
                allocator: alloc,
            },
        ))
    }

    pub fn new_uninit_slice_in(
        len: usize,
        alloc: A,
    ) -> (
        ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
        ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
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
            ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A>,
            ArcReaderLock<[MaybeUninit<T>], A>,
        ),
        AllocError,
    > {
        Self::try_new_zeroed_slice_in_with_layout(len, alloc).map_err(|err| match err {
            TryAllocError::Layout(_) => AllocError,
            TryAllocError::Alloc { err, .. } => err,
        })
    }
}

impl<T, U, A: Allocator> ArcMappedRwLock<MaybeUninit<T>, MaybeUninit<U>, A> {
    pub const unsafe fn assume_init(self) -> ArcMappedRwLock<T, U, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        ArcMappedRwLock {
            lock: MappedRwLock {
                inner: lock.inner.cast(),
                subfield: lock.subfield.cast(),
            },
            allocator,
        }
    }
}

impl<T, A: Allocator> ArcMappedRwLock<MaybeUninit<T>, [MaybeUninit<T>], A> {
    pub const unsafe fn assume_init(self) -> ArcMappedRwLock<T, [T], A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        ArcMappedRwLock {
            lock: MappedRwLock {
                inner: {
                    let (ptr, len) = lock.inner.to_raw_parts();
                    NonNull::from_raw_parts(ptr, len)
                },
                subfield: lock.subfield.cast(),
            },
            allocator,
        }
    }
}

impl<T, A: Allocator> ArcMappedRwLock<[MaybeUninit<T>], [MaybeUninit<T>], A> {
    pub const unsafe fn assume_init(self) -> ArcMappedRwLock<[T], [T], A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        ArcMappedRwLock {
            lock: MappedRwLock {
                inner: {
                    let (ptr, len) = lock.inner.to_raw_parts();
                    NonNull::from_raw_parts(ptr, len)
                },
                subfield: {
                    let (ptr, len) = lock.subfield.to_raw_parts();
                    NonNull::from_raw_parts(ptr, len)
                },
            },
            allocator,
        }
    }
}

impl<T: ?Sized, U: ?Sized, A: Allocator> DerefMut for ArcMappedRwLock<T, U, A> {
    fn deref_mut(&mut self) -> &mut MappedRwLock<T, U> {
        &mut self.lock
    }
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

impl<T, U, V, A> CoerceUnsized<ArcMappedRwLock<V, U, A>> for ArcMappedRwLock<T, U, A>
where
    T: Unsize<V> + ?Sized,
    U: ?Sized,
    A: Allocator,
{
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

impl<T: ?Sized, U: ?Sized, A: AllocatorClone> Clone for ArcMappedRwLock<T, U, A> {
    fn clone(&self) -> Self {
        if unsafe {
            // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
            InnerArc::increment_shared_counter(
                // SAFETY: `lock.inner` has been allocated as a part of an `InnerArc`.
                InnerArc::from_lock(self.lock.inner).0,
                Ordering::Relaxed,
            )
        } {
            process::abort();
        }
        Self {
            lock: MappedRwLock { ..self.lock },
            allocator: self.allocator.clone(),
        }
    }
}
