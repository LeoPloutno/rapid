use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy, Debug)]
pub struct MapOutsideWhole<T, U> {
    map: T,
    whole: U,
}

impl<T, U> MapOutsideWhole<T, U> {
    pub fn map_map<V, F>(self, f: F) -> MapOutsideWhole<V, U>
    where
        F: FnOnce(T) -> V,
    {
        MapOutsideWhole {
            map: f(self.map),
            whole: self.whole,
        }
    }

    pub fn try_map_map<V, E, F>(self, f: F) -> Result<MapOutsideWhole<V, U>, E>
    where
        F: FnOnce(T) -> Result<V, E>,
    {
        Ok(MapOutsideWhole {
            map: f(self.map)?,
            whole: self.whole,
        })
    }

    pub fn map_whole<V, F>(self, f: F) -> MapOutsideWhole<T, V>
    where
        F: FnOnce(U) -> V,
    {
        MapOutsideWhole {
            map: self.map,
            whole: f(self.whole),
        }
    }

    pub fn try_map_whole<V, E, F>(self, f: F) -> Result<MapOutsideWhole<T, V>, E>
    where
        F: FnOnce(U) -> Result<V, E>,
    {
        Ok(MapOutsideWhole {
            map: self.map,
            whole: f(self.whole)?,
        })
    }

    pub fn as_map(&self) -> &T::Target
    where
        T: Deref,
    {
        &*self.map
    }

    pub fn as_mut_map(&mut self) -> &mut T::Target
    where
        T: DerefMut,
    {
        &mut *self.map
    }

    pub fn as_whole(&self) -> &U::Target
    where
        U: Deref,
    {
        &*self.whole
    }

    pub fn as_mut_whole(&mut self) -> &mut U::Target
    where
        U: DerefMut,
    {
        &mut *self.whole
    }

    pub fn as_ref(&self) -> MapOutsideWhole<&T::Target, &U::Target>
    where
        T: Deref,
        U: Deref,
    {
        MapOutsideWhole {
            map: &*self.map,
            whole: &*self.whole,
        }
    }

    pub fn as_map_ref(&self) -> MapOutsideWhole<&T::Target, U>
    where
        T: Deref,
        U: Clone,
    {
        MapOutsideWhole {
            map: &*self.map,
            whole: self.whole.clone(),
        }
    }

    pub fn as_whole_ref(&self) -> MapOutsideWhole<T, &U::Target>
    where
        T: Clone,
        U: Deref,
    {
        MapOutsideWhole {
            map: self.map.clone(),
            whole: &*self.whole,
        }
    }

    pub fn as_mut(&mut self) -> MapOutsideWhole<&mut T::Target, &mut U::Target>
    where
        T: DerefMut,
        U: DerefMut,
    {
        MapOutsideWhole {
            map: &mut *self.map,
            whole: &mut *self.whole,
        }
    }

    pub fn as_map_mut(&mut self) -> MapOutsideWhole<&mut T::Target, U>
    where
        T: DerefMut,
        U: Clone,
    {
        MapOutsideWhole {
            map: &mut *self.map,
            whole: self.whole.clone(),
        }
    }

    pub fn as_whole_mut(&mut self) -> MapOutsideWhole<T, &mut U::Target>
    where
        T: Clone,
        U: DerefMut,
    {
        MapOutsideWhole {
            map: self.map.clone(),
            whole: &mut *self.whole,
        }
    }
}

impl<T: Deref, U> Deref for MapOutsideWhole<T, U> {
    type Target = T::Target;

    /// Equivalent to [`MapOutsideWhole::as_map`].
    fn deref(&self) -> &Self::Target {
        &*self.map
    }
}

impl<T: DerefMut, U> DerefMut for MapOutsideWhole<T, U> {
    /// Equivalent to [`MapOutsideWhole::as_map_mut`].
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.map
    }
}

impl<T: Deref, U> AsRef<T::Target> for MapOutsideWhole<T, U> {
    /// Equivalent to [`MapOutsideWhole::as_map`].
    fn as_ref(&self) -> &T::Target {
        &*self.map
    }
}

impl<T: DerefMut, U> AsMut<T::Target> for MapOutsideWhole<T, U> {
    /// Equivalent to [`MapOutsideWhole::as_map_mut`].
    fn as_mut(&mut self) -> &mut T::Target {
        &mut *self.map
    }
}

impl<T, U, V> From<MapOutsideWhole<T, MapOutsideWhole<U, V>>> for MapOutsideWhole<T, U> {
    fn from(value: MapOutsideWhole<T, MapOutsideWhole<U, V>>) -> Self {
        Self {
            map: value.map,
            whole: value.whole.map,
        }
    }
}

impl<T, U, V> From<MapOutsideWhole<MapOutsideWhole<T, U>, V>> for MapOutsideWhole<U, V> {
    fn from(value: MapOutsideWhole<MapOutsideWhole<T, U>, V>) -> Self {
        Self {
            map: value.map.whole,
            whole: value.whole,
        }
    }
}
