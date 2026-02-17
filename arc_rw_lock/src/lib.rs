#![allow(dead_code)]
#![feature(allocator_api, ptr_metadata, layout_for_ptr, sync_nonpoison)]

use std::{
    alloc::{Allocator, Global},
    marker::PhantomData,
    ptr::NonNull,
};

mod alloc;
mod arc;
mod inner;
mod lock;
mod rw_lock;
mod slice;
mod unique_arc;

#[cold]
fn unlikely<T>(value: T) -> T {
    value
}

#[cold]
fn cold_path() {}

pub use lock::{MappedRwLock, MappedRwLockReadWholeGuard, MappedRwLockWriteGuard};

pub use arc::{ArcMappedRwLock, UniqueArcMappedRwLock};

pub type ElementRwLock<T> = MappedRwLock<T, [T]>;

pub type SliceRwLock<T> = MappedRwLock<[T], [T]>;

pub type ArcElementRwLock<T, A = Global> = ArcMappedRwLock<T, [T], A>;

pub type ArcSliceRwLock<T, A = Global> = ArcMappedRwLock<[T], [T], A>;

pub type UniqueArcElementRwLock<T, A = Global> = UniqueArcMappedRwLock<T, [T], A>;

pub type UniqueArcSliceRwLock<T, A = Global> = UniqueArcMappedRwLock<[T], [T], A>;
