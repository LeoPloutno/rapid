use lib::{
    core::{AtomType, AtomTypeCharacteristics, GroupInTypeInImageInSystem, Vector},
    output::{StepStream, VectorsStream},
};
use macros::warn;
use std::{
    error::Error,
    fmt::{Debug, Display, Formatter, LowerExp, Result as FmtResult},
    io::{Error as IOError, Write},
};

#[derive(Debug)]
pub enum XYZError {
    IO(IOError),
    Sync,
}

impl From<IOError> for XYZError {
    fn from(value: IOError) -> Self {
        Self::IO(value)
    }
}

impl Display for XYZError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::IO(err) => write!(f, "io error: {}", err),
            Self::Sync => write!(f, "synchronization error"),
        }
    }
}

impl Error for XYZError {}

pub struct XYZFileStream<W, const N: usize> {
    stream: W,
    atoms: usize,
    precision: usize,
}

impl<W, const N: usize> XYZFileStream<W, N> {
    pub fn new(stream: W, atoms: usize, precision: usize) -> Self {
        Self {
            stream,
            atoms,
            precision,
        }
    }
}

impl<W: Write, const N: usize> StepStream for XYZFileStream<W, N> {
    type Error = XYZError;

    fn write_prelude(&mut self, step: usize) -> Result<(), Self::Error> {
        write!(self.stream, "# step: {}\n{}", step, self.atoms)?;
        Ok(())
    }

    fn new_line(&mut self) -> Result<(), Self::Error> {
        write!(self.stream, "\n")?;
        Ok(())
    }
}

impl<T, V, W> VectorsStream<T, V> for XYZFileStream<W, 2>
where
    T: LowerExp,
    V: Vector<2, Element = T>,
    W: Write,
{
    fn write_vectors(&mut self, vectors: &GroupInTypeInImageInSystem<T, V>) -> Result<(), Self::Error> {
        warn!("optimize");
        for AtomType {
            characteristics: AtomTypeCharacteristics { name, .. },
            lock,
        } in vectors.whole.as_whole().iter()
        {
            for group_lock in lock.read().map_err(|_| XYZError::Sync)?.iter() {
                for atom in group_lock.read() {
                    let [x, y] = atom.as_array();
                    write!(self.stream, "\n{1} {2:.0$e} {3:.0$e}", self.precision, name, *x, *y)?;
                }
            }
        }
        Ok(())
    }
}

impl<T, V, W> VectorsStream<T, V> for XYZFileStream<W, 3>
where
    T: LowerExp,
    V: Vector<3, Element = T>,
    W: Write,
{
    fn write_vectors(&mut self, vectors: &GroupInTypeInImageInSystem<T, V>) -> Result<(), Self::Error> {
        warn!("optimize");
        for AtomType {
            characteristics: AtomTypeCharacteristics { name, .. },
            lock,
        } in vectors.whole.as_whole().iter()
        {
            for group_lock in lock.read().map_err(|_| XYZError::Sync)?.iter() {
                for atom in group_lock.read() {
                    let [x, y, z] = atom.as_array();
                    write!(
                        self.stream,
                        "\n{1} {2:.0$e} {3:.0$e} {4:.0$e}",
                        self.precision, name, *x, *y, *z
                    )?;
                }
            }
        }
        Ok(())
    }
}
