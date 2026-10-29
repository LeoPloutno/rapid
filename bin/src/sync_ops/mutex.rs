use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    ops::{Add, Mul},
    sync::{Condvar, Mutex},
};

use lib::core::{
    marker::NonInfallible,
    sync_ops::{SyncAddReceiver, SyncAddSender, SyncMulReceiver, SyncMulSender},
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

pub struct MutexSyncOp<T> {
    mutex: Mutex<(usize, Option<T>)>,
    counter: usize,
    condvar: Condvar,
}

impl<T> MutexSyncOp<T> {
    pub fn new(senders: usize) -> Self {
        Self {
            mutex: Mutex::new((senders, None)),
            counter: senders,
            condvar: Condvar::new(),
        }
    }
}

impl<T: Add<Output = T> + Clone> SyncAddSender<T> for MutexSyncOp<T> {
    type Error = SyncError;

    fn send(&mut self, value: T) -> Result<(), Self::Error> {
        let mut guard = self.mutex.lock().map_err(|_| SyncError)?;
        if let Some(shared) = &mut (*guard).1 {
            *shared = shared.clone() + value;
        } else {
            guard.1 = Some(value);
        }
        let prev_counter = guard.0;
        guard.0 = prev_counter - 1;
        if prev_counter == 1 {
            drop(guard);
            self.condvar.notify_one();
        }
        Ok(())
    }

    fn send_empty(&mut self) -> Result<(), Self::Error> {
        let mut guard = self.mutex.lock().map_err(|_| SyncError)?;
        let prev_counter = guard.0;
        guard.0 = prev_counter - 1;
        if prev_counter == 1 {
            drop(guard);
            self.condvar.notify_one();
        }
        Ok(())
    }
}

impl<T: Clone> SyncAddReceiver<T> for MutexSyncOp<T> {
    type Error = SyncError;

    fn recv_sum(&mut self) -> Result<Option<T>, Self::Error> {
        let mut guard = self
            .condvar
            .wait_while(self.mutex.lock().map_err(|_| SyncError)?, |&mut (counter, _)| {
                counter != 0
            })
            .map_err(|_| SyncError)?;
        guard.0 = self.counter;
        Ok(guard.1.clone())
    }
}

impl<T: Mul<Output = T> + Clone> SyncMulSender<T> for MutexSyncOp<T> {
    type Error = SyncError;

    fn send(&mut self, value: T) -> Result<(), Self::Error> {
        let mut guard = self.mutex.lock().map_err(|_| SyncError)?;
        if let Some(shared) = &mut (*guard).1 {
            *shared = shared.clone() * value;
        }
        let prev_counter = guard.0;
        guard.0 = prev_counter - 1;
        if prev_counter == 1 {
            drop(guard);
            self.condvar.notify_one();
        }
        Ok(())
    }

    fn send_empty(&mut self) -> Result<(), Self::Error> {
        let mut guard = self.mutex.lock().map_err(|_| SyncError)?;
        let prev_counter = guard.0;
        guard.0 = prev_counter - 1;
        if prev_counter == 1 {
            drop(guard);
            self.condvar.notify_one();
        }
        Ok(())
    }
}

impl<T: Clone> SyncMulReceiver<T> for MutexSyncOp<T> {
    type Error = SyncError;

    fn recv_prod(&mut self) -> Result<Option<T>, Self::Error> {
        let mut guard = self
            .condvar
            .wait_while(self.mutex.lock().map_err(|_| SyncError)?, |&mut (counter, _)| {
                counter != 0
            })
            .map_err(|_| SyncError)?;
        guard.0 = self.counter;
        Ok(guard.1.clone())
    }
}
