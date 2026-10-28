use lib::output::{StepStream, ValuesStream};
use macros::warn;
use std::{fmt::Display, io::Error as IOError, io::Write};

pub struct DefaultStream<W> {
    stream: W,
    step_width: usize,
    padding: usize,
}

impl<W> DefaultStream<W> {
    pub fn new(stream: W, max_step: usize, padding: usize) -> Self {
        Self {
            stream,
            step_width: max_step.ilog10() as _,
            padding,
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
    T: Display,
    W: Write,
{
    fn write_value(&mut self, value: T) -> Result<(), Self::Error> {
        warn!("optimize");
        write!(self.stream, "{0:1$}{2}", ' ', self.padding, value)
    }
}
