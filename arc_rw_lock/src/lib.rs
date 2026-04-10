#![allow(dead_code)]
#![feature(allocator_api, ptr_metadata, layout_for_ptr, sync_nonpoison)]

use std::alloc::Global;

mod alloc;
mod arc;
pub use arc::{ArcMappedRwLock, ArcReaderLock, UniqueArcMappedRwLock};
mod lock;
pub use lock::{MappedRwLock, MappedRwLockGuard, ReaderLock, ReaderLockGuard};
mod slice;
mod unique_arc;

#[cold]
fn unlikely<T>(value: T) -> T {
    value
}

#[cold]
fn cold_path() {}

pub type ElementRwLock<T> = MappedRwLock<T, [T]>;

pub type SliceRwLock<T> = MappedRwLock<[T], [T]>;

pub type SliceReaderLock<T> = ReaderLock<[T]>;

pub type ArcElementRwLock<T, A = Global> = ArcMappedRwLock<T, [T], A>;

pub type ArcSliceRwLock<T, A = Global> = ArcMappedRwLock<[T], [T], A>;

pub type ArcSliceReaderLock<T, A = Global> = ArcReaderLock<[T], A>;

pub type UniqueArcElementRwLock<T, A = Global> = UniqueArcMappedRwLock<T, [T], A>;

pub type UniqueArcSliceRwLock<T, A = Global> = UniqueArcMappedRwLock<[T], [T], A>;
