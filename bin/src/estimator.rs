use lib::core::marker::NonInfallible;
use std::{
    convert::Infallible,
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

pub mod classical;
pub mod quantum;

#[derive(Debug)]
pub struct EstimatorsError(Box<dyn Error + Send + Sync>);

impl From<Infallible> for EstimatorsError {
    fn from(value: Infallible) -> Self {
        value
    }
}

impl<E> From<E> for EstimatorsError
where
    E: Error + Send + Sync + NonInfallible + 'static,
{
    fn from(value: E) -> Self {
        Self(value.into())
    }
}

impl Display for EstimatorsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

impl Error for EstimatorsError {}
