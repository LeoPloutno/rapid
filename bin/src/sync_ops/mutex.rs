use lib::core::{
    marker::NonInfallible,
    sync_ops::{SyncAddReceiver, SyncAddSender, SyncMulReceiver, SyncMulSender},
};
use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    ops::{AddAssign, MulAssign},
    sync::Arc,
    sync::{Condvar, Mutex},
};

#[derive(Debug)]
pub struct SyncError;

impl Display for SyncError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "synchronization failed")
    }
}

impl Error for SyncError {}

impl NonInfallible for SyncError {}

struct Data<T> {
    remaining: usize,
    data: Option<T>,
}

struct InnerMutexSync<T> {
    lock: Mutex<Data<T>>,
    condvar: Condvar,
}

pub struct MutexSyncReceiver<T> {
    inner: Arc<InnerMutexSync<T>>,
    senders: usize,
}

impl<T: Clone> SyncAddReceiver<T> for MutexSyncReceiver<T> {
    type Error = SyncError;

    fn recv_sum(&mut self) -> Result<Option<T>, Self::Error> {
        let mut guard = self
            .inner
            .condvar
            .wait_while(
                self.inner.lock.lock().map_err(|_| SyncError)?,
                |&mut Data { remaining, .. }| remaining != 0,
            )
            .map_err(|_| SyncError)?;
        guard.remaining = self.senders;
        Ok(guard.data.take())
    }
}

impl<T> SyncMulReceiver<T> for MutexSyncReceiver<T> {
    type Error = SyncError;

    fn recv_prod(&mut self) -> Result<Option<T>, Self::Error> {
        let mut guard = self
            .inner
            .condvar
            .wait_while(
                self.inner.lock.lock().map_err(|_| SyncError)?,
                |&mut Data { remaining, .. }| remaining != 0,
            )
            .map_err(|_| SyncError)?;
        guard.remaining = self.senders;
        Ok(guard.data.take())
    }
}

pub struct MutexSyncSender<T>(Arc<InnerMutexSync<T>>);

impl<T: AddAssign<T>> SyncAddSender<T> for MutexSyncSender<T> {
    type Error = SyncError;

    fn send(&mut self, value: T) -> Result<(), Self::Error> {
        let mut guard = self.0.lock.lock().map_err(|_| SyncError)?;
        if let Some(shared) = &mut (*guard).data {
            *shared += value;
        } else {
            guard.data = Some(value);
        }
        let prev_counter = guard.remaining;
        guard.remaining = prev_counter - 1;
        if prev_counter == 1 {
            drop(guard);
            self.0.condvar.notify_one();
        }
        Ok(())
    }

    fn send_empty(&mut self) -> Result<(), Self::Error> {
        let mut guard = self.0.lock.lock().map_err(|_| SyncError)?;
        let prev_counter = guard.remaining;
        guard.remaining = prev_counter - 1;
        if prev_counter == 1 {
            drop(guard);
            self.0.condvar.notify_one();
        }
        Ok(())
    }
}

impl<T: MulAssign<T>> SyncMulSender<T> for MutexSyncSender<T> {
    type Error = SyncError;

    fn send(&mut self, value: T) -> Result<(), Self::Error> {
        let mut guard = self.0.lock.lock().map_err(|_| SyncError)?;
        if let Some(shared) = &mut (*guard).data {
            *shared *= value;
        } else {
            guard.data = Some(value);
        }
        let prev_counter = guard.remaining;
        guard.remaining = prev_counter - 1;
        if prev_counter == 1 {
            drop(guard);
            self.0.condvar.notify_one();
        }
        Ok(())
    }

    fn send_empty(&mut self) -> Result<(), Self::Error> {
        let mut guard = self.0.lock.lock().map_err(|_| SyncError)?;
        let prev_counter = guard.remaining;
        guard.remaining = prev_counter - 1;
        if prev_counter == 1 {
            drop(guard);
            self.0.condvar.notify_one();
        }
        Ok(())
    }
}

pub fn mutex_sync_op<T>(senders: usize) -> (MutexSyncReceiver<T>, Box<[MutexSyncSender<T>]>) {
    let inner = Arc::new(InnerMutexSync {
        lock: Mutex::new(Data {
            remaining: senders,
            data: None,
        }),
        condvar: Condvar::new(),
    });
    let mut boxed_senders = Box::new_uninit_slice(senders);
    for sender in boxed_senders.iter_mut() {
        sender.write(MutexSyncSender(inner.clone()));
    }
    // SAFETY: Initialized all elements above.
    (MutexSyncReceiver { inner, senders }, unsafe {
        boxed_senders.assume_init()
    })
}
