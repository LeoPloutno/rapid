#![allow(dead_code)]
#![feature(allocator_api, ptr_metadata, layout_for_ptr, sync_nonpoison)]

mod alloc;
mod arc;
mod inner;
mod lock;
mod rw_lock;
mod unique_arc;

use std::{
    alloc::{Allocator, Global},
    marker::PhantomData,
    ptr::NonNull,
};

pub struct ArcRwLock<T: ?Sized, A: Allocator = Global> {
    _phantom: PhantomData<(NonNull<T>, A)>,
}

pub struct UniqueArcRwLock<T: ?Sized, A: Allocator = Global> {
    _phantom: PhantomData<(NonNull<T>, A)>,
}

pub use lock::{MappedRwLock, MappedRwLockReadWholeGuard, MappedRwLockWriteGuard};

pub use arc::{ArcMappedRwLock, UniqueArcMappedRwLock};

pub type ElementRwLock<T> = MappedRwLock<T, [T]>;

pub type SliceRwLock<T> = MappedRwLock<[T], [T]>;

pub type ArcElementRwLock<T, A = Global> = ArcMappedRwLock<T, [T], A>;

pub type ArcSliceRwLock<T, A = Global> = ArcMappedRwLock<[T], [T], A>;

pub type UniqueArcElementRwLock<T, A = Global> = UniqueArcMappedRwLock<T, [T], A>;

pub type UniqueArcSliceRwLock<T, A = Global> = UniqueArcMappedRwLock<[T], [T], A>;
