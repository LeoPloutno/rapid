pub trait SyncAddSend<T> {
    type Error;

    /// Sends `value` to the adder.
    fn send(&mut self, value: T) -> Result<(), Self::Error>;

    /// Sends an empty message to the adder.
    fn send_empty(&mut self) -> Result<(), Self::Error>;
}

pub trait SyncAddRecv<T> {
    type Error;

    /// Recieves the sum of all non-empty messages.
    fn recieve_sum(&mut self) -> Result<Option<T>, Self::Error>;
}

pub trait SyncMulSend<T> {
    type Error;

    /// Sends `value` to the multiplier.
    fn send(&mut self, value: T) -> Result<(), Self::Error>;

    /// Sends an empty message to the multiplier.
    fn send_empty(&mut self) -> Result<(), Self::Error>;
}

pub trait SyncMulRecv<T> {
    type Error;

    /// Recieves the product of all non-empty messages.
    fn recieve_prod(&mut self) -> Result<Option<T>, Self::Error>;
}
