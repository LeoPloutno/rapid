use lib::output::{StepStream, ValuesStream};
use macros::warn;
use std::{
    fmt::{Display, LowerExp},
    io::{Error as IOError, Write},
};

pub struct DefaultStream<W> {
    stream: W,
    step_width: usize,
    padding: usize,
    precision: usize,
}

impl<W> DefaultStream<W> {
    pub fn new(stream: W, max_step: usize, padding: usize, precision: usize) -> Self {
        Self {
            stream,
            step_width: max_step.ilog10() as _,
            padding,
            precision,
        }
    }
}

impl<W: Write> StepStream for DefaultStream<W> {
    type Error = IOError;

    fn write_prelude(&mut self, step: usize) -> Result<(), Self::Error> {
        warn!("optimize");
        write!(self.stream, "{:1$}", step, self.step_width + self.padding)
    }

    fn new_line(&mut self) -> Result<(), Self::Error> {
        warn!("optimmize");
        write!(self.stream, "\n")
    }
}

impl<T, W> ValuesStream<T> for DefaultStream<W>
where
    T: Display + LowerExp,
    W: Write,
{
    fn write_value(&mut self, value: T) -> Result<(), Self::Error> {
        warn!("optimize");
        write!(self.stream, "{0:1$}{3:+.2$e}", ' ', self.padding, self.precision, value)
    }
}
