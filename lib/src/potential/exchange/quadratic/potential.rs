use crate::marker::{InnerIsLeading, InnerIsTrailing};

use super::{
    super::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential},
    transform::{InnerNormalModesTransform, LeadingNormalModesTransform, TrailingNormalModesTransform},
};

/// A trait for leading exchange potentials that may be expanded to second order.
pub trait LeadingQuadraticExpansionExchangePotential<'a, T, V> {
    /// The transformation that yields the normal modes such that
    /// the second order term is the sum over all modes squared times their
    /// respective eigenvalues.
    type QuadraticPotential: LeadingNormalModesTransform<T, V>;

    /// The term left after the expansion to second order.
    /// Contains interactions of third order and higher.
    type ResiduePotential: LeadingExchangePotential<T, V>;

    /// Treats `self` as a sum of harmonic oscillators - the normal modes -
    /// and a residual term of third order and beyond.
    fn as_quadratic_expansion(&'a mut self) -> (Self::QuadraticPotential, Self::ResiduePotential);
}

/// A trait for inner exchange potentials that may be expanded to second order.
pub trait InnerQuadraticExpansionExchangePotential<'a, T, V> {
    /// The transformation that yields the normal modes such that
    /// the second order term is the sum over all modes squared times their
    /// respective eigenvalues.
    type QuadraticPotential: InnerNormalModesTransform<T, V>;

    /// The term left after the expansion to second order.
    /// Contains interactions of third order and higher.
    type ResiduePotential: InnerExchangePotential<T, V>;

    /// Treats `self` as a sum of harmonic oscillators - the normal modes -
    /// and a residual term of third order and beyond.
    fn as_quadratic_expansion(&'a mut self) -> (Self::QuadraticPotential, Self::ResiduePotential);
}

/// A trait for trailing exchange potentials that may be expanded to second order.
pub trait TrailingQuadraticExpansionExchangePotential<'a, T, V> {
    /// The transformation that yields the normal modes such that
    /// the second order term is the sum over all modes squared times their
    /// respective eigenvalues.
    type QuadraticPotential: TrailingNormalModesTransform<T, V>;

    /// The term left after the expansion to second order.
    /// Contains interactions of third order and higher.
    type ResiduePotential: TrailingExchangePotential<T, V>;

    /// Treats `self` as a sum of harmonic oscillators - the normal modes -
    /// and a residual term of third order and beyond.
    fn as_quadratic_expansion(&'a mut self) -> (Self::QuadraticPotential, Self::ResiduePotential);
}

impl<'a, T, V, U> LeadingQuadraticExpansionExchangePotential<'a, T, V> for U
where
    U: InnerQuadraticExpansionExchangePotential<'a, T, V> + InnerIsLeading,
    U::QuadraticPotential: LeadingNormalModesTransform<T, V>,
    U::ResiduePotential: LeadingExchangePotential<T, V>,
{
    type QuadraticPotential = <Self as InnerQuadraticExpansionExchangePotential<'a, T, V>>::QuadraticPotential;

    type ResiduePotential = <Self as InnerQuadraticExpansionExchangePotential<'a, T, V>>::ResiduePotential;

    fn as_quadratic_expansion(&'a mut self) -> (Self::QuadraticPotential, Self::ResiduePotential) {
        InnerQuadraticExpansionExchangePotential::as_quadratic_expansion(self)
    }
}

impl<'a, T, V, U> TrailingQuadraticExpansionExchangePotential<'a, T, V> for U
where
    U: InnerQuadraticExpansionExchangePotential<'a, T, V> + InnerIsTrailing,
    U::QuadraticPotential: TrailingNormalModesTransform<T, V>,
    U::ResiduePotential: TrailingExchangePotential<T, V>,
{
    type QuadraticPotential = <Self as InnerQuadraticExpansionExchangePotential<'a, T, V>>::QuadraticPotential;

    type ResiduePotential = <Self as InnerQuadraticExpansionExchangePotential<'a, T, V>>::ResiduePotential;

    fn as_quadratic_expansion(&'a mut self) -> (Self::QuadraticPotential, Self::ResiduePotential) {
        InnerQuadraticExpansionExchangePotential::as_quadratic_expansion(self)
    }
}
