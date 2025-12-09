#![feature(allocator_api, ptr_metadata)]

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

pub struct MappedRwLock<T: ?Sized, U: ?Sized> {
    _phantom: PhantomData<(NonNull<T>, NonNull<U>)>,
}

pub struct ArcMappedRwLock<T: ?Sized, U: ?Sized = dyn 'static + Send + Sync, A: Allocator = Global>
{
    _phantom: PhantomData<(NonNull<T>, NonNull<U>, A)>,
}

pub struct UniqueArcMappedRwLock<
    T: ?Sized,
    U: ?Sized = dyn 'static + Send + Sync,
    A: Allocator = Global,
> {
    _phantom: PhantomData<(NonNull<T>, NonNull<U>, A)>,
}

pub type ElementRwLock<T> = MappedRwLock<T, [T]>;

pub type SliceRwLock<T> = MappedRwLock<[T], [T]>;

pub type ArcElementRwLock<T, A = Global> = ArcMappedRwLock<T, [T], A>;

pub type ArcSliceRwLock<T, A = Global> = ArcMappedRwLock<[T], [T], A>;

pub type UniqueArcElementRwLock<T, A = Global> = ArcMappedRwLock<T, [T], A>;

pub type UniqueArcSliceRwLock<T, A = Global> = ArcMappedRwLock<[T], [T], A>;
