//! Traits for parallelized calculations.

use std::{
    error::Error,
    ops::{Deref, DerefMut},
};

/// A trait for objects which add up values and send the sum to a `SyncAddReceiver`.
pub trait SyncAddSender<T> {
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Sends `value` to the adder.
    fn send(&mut self, value: T) -> Result<(), Self::Error>;

    /// Sends an empty message to the adder.
    fn send_empty(&mut self) -> Result<(), Self::Error>;
}

impl<T, S> SyncAddSender<T> for S
where
    S: DerefMut<Target: SyncAddSender<T>> + ?Sized,
{
    type Error = <<Self as Deref>::Target as SyncAddSender<T>>::Error;

    #[inline]
    fn send(&mut self, value: T) -> Result<(), Self::Error> {
        (*self).send(value)
    }

    #[inline]
    fn send_empty(&mut self) -> Result<(), Self::Error> {
        (*self).send_empty()
    }
}

/// A trait for objects which receive the sum calculated by `SyncAddSender`s.
pub trait SyncAddReceiver<T> {
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Receives the sum of all non-empty messages.
    fn recv_sum(&mut self) -> Result<Option<T>, Self::Error>;
}

impl<T, R> SyncAddReceiver<T> for R
where
    R: DerefMut<Target: SyncAddReceiver<T>> + ?Sized,
{
    type Error = <<Self as Deref>::Target as SyncAddReceiver<T>>::Error;

    #[inline]
    fn recv_sum(&mut self) -> Result<Option<T>, Self::Error> {
        (*self).recv_sum()
    }
}

/// A trait for objects which multiply values and send the product to a `SyncAddReceiver`.
pub trait SyncMulSender<T> {
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Sends `value` to the multiplier.
    fn send(&mut self, value: T) -> Result<(), Self::Error>;

    /// Sends an empty message to the multiplier.
    fn send_empty(&mut self) -> Result<(), Self::Error>;
}

impl<T, S> SyncMulSender<T> for S
where
    S: DerefMut<Target: SyncMulSender<T>> + ?Sized,
{
    type Error = <<Self as Deref>::Target as SyncMulSender<T>>::Error;

    #[inline]
    fn send(&mut self, value: T) -> Result<(), Self::Error> {
        (*self).send(value)
    }

    #[inline]
    fn send_empty(&mut self) -> Result<(), Self::Error> {
        (*self).send_empty()
    }
}

/// A trait for objects which receive the product calculated by `SyncAddSender`s.
pub trait SyncMulReceiver<T> {
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Receives the product of all non-empty messages.
    fn recv_prod(&mut self) -> Result<Option<T>, Self::Error>;
}

impl<T, R> SyncMulReceiver<T> for R
where
    R: DerefMut<Target: SyncMulReceiver<T>> + ?Sized,
{
    type Error = <<Self as Deref>::Target as SyncMulReceiver<T>>::Error;

    #[inline]
    fn recv_prod(&mut self) -> Result<Option<T>, Self::Error> {
        (*self).recv_prod()
    }
}
