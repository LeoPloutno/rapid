//! Traits for parallelized calculations.

/// A trait for objects which add up values and send the sum to a `SyncAddReciever`.
pub trait SyncAddSender<T> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Sends `value` to the adder.
    fn send(&mut self, value: T) -> Result<(), Self::Error>;

    /// Sends an empty message to the adder.
    fn send_empty(&mut self) -> Result<(), Self::Error>;
}

/// A trait for objects which recieve the sum calculated by `SyncAddSender`s.
pub trait SyncAddReciever<T> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Recieves the sum of all non-empty messages.
    fn recieve_sum(&mut self) -> Result<Option<T>, Self::Error>;
}

/// A trait for objects which multiply values and send the product to a `SyncAddReciever`.
pub trait SyncMulSender<T> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Sends `value` to the multiplier.
    fn send(&mut self, value: T) -> Result<(), Self::Error>;

    /// Sends an empty message to the multiplier.
    fn send_empty(&mut self) -> Result<(), Self::Error>;
}

/// A trait for objects which recieve the product calculated by `SyncAddSender`s.
pub trait SyncMulReciever<T> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Recieves the product of all non-empty messages.
    fn recieve_prod(&mut self) -> Result<Option<T>, Self::Error>;
}
