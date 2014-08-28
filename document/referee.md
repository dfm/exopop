# Referee Report

> This paper describes a statistical analysis of the Kepler exoplanet catalog,
> with particular emphasis on estimating the fraction of stars having Earth
> analogs, that is, planets of roughly an Earth radius orbiting at roughly 1 AU.
> The subject is an important one, the discussion is provocative and interesting,
> and the statistical analysis is significantly more general and rigorous than in
> any prior attempt to estimate the distribution of the Kepler planets. Thus the
> paper certainly deserves to be published.
>
> Nevertheless, I am concerned that the paper will not have the impact that it
> should, for several reasons, and I urge the authors to consider the following
> suggestions to improve the presentation and the clarity of the conclusions:
>
> 1. The Astrophysical Journal is, after all, a journal for physicists and
> astronomers rather than statisticians, and the heavy use of statistical jargon
> (hierarchical inference, censored occurrence rate, sufficient statistics,
> conditional independence, importance sampling, support, hyperparameters) and
> lengthy presentation of the statistical method will likely turn off many of the
> potential consumers of the new techniques described here. Could not the main
> text contain a briefer description of the methods, with details placed in an
> Appendix? Or could the reader not be told to skip to a summary at the end of
> Section 3? To put it more bluntly, is the paper written so a typical grad
> student working on Kepler will both understand how to implement the method and
> recognize that this is better than the methods he or she has been using up to
> now?

We have taken this criticism to heart and we hope that

> 2. The restriction of the Petigura et al. catalog to the highest S/N planet
> candidate in multi-planet systems is important. However, it is brought up in
> the first paragraph on p. 4, independently in the second paragraph on the same
> page, again (twice) on p. 8; again on p. 19; and again on pp. 20-21. Shouldn't
> these be collected into a single discussion?

Agreed. We have removed most of these references, leaving one in the
introduction and expanding the discussion in the conclusion section.

> 3. I like to think that I'm reasonably familiar with Bayesian statistics, but I
> was unable to figure out what the parameter \alpha was and what role it played
> in the derivations.

This is definitely a subtle and important point. We have added a few sentences
to that section to try to make it clearer. The key point is that \alpha
references the prior that the author of the catalog chose. This will normally
be uniform or something else uninformative and the point of hierarchical
inference is to update this initial distribution.

> 4. The discussion of the inverse-detection-efficiency method is misleading, in
> that it conflates two separate approximations: the first is to treat each data
> point as a delta function in the probability distribution, inversely weighted
> by the detection efficiency; the second is to smooth these delta functions over
> a finite bin. These approximations are quite different and it's not clear
> whether the authors' objection to this method arises from the first
> approximation, or the second, or both.

As mentioned above, we have moved this discussion to the appendix so that it
is less of a focus. The point of the discussion is meant to be: if you want to
neglect observational uncertainties and model the occurrence rate as a step
function, there is a simple analytic result but it's not the same as
inverse-detection-efficiency! If the uncertainties are non-negligible then the
solution is no longer analytic and you'll need to do something like our method.
We have expanded the discussion in the appendix in an attempt to make this
clearer.

> 5. The value derived in this paper for Gamma_Earth, 0.017 +0.018/-0.009,
> differs from the value given by Petigura et al., 0.119 +0.035/-0.046 , by more
> than 2 sigma, even though the two analyses use exactly the same data. This
> discrepancy is surprising and important, and one of the primary goals of the
> paper should be to explain how it arises, and what, if anything, Petigura et
> al. should have done differently. The paper fails at this goal. For example, at
> various times in reading the paper I concluded that the problem with Petigura
> et al. was that:
>
> (a) they extrapolate by assuming that the distribution per unit log period is
> flat, rather than modeling the period and radius distribution as a Gaussian
> process;
>
> (b) they use the inverse-detection-efficiency method rather than a Bayesian
> approach;
>
> (c) they did not incorporate radius errors in their statistical analysis, even
> though they derived these in their catalogs (at least I think this is true from
> reading Petigura et al.; the discussion in the first paragraph of section 4
> suggests otherwise).
>
> Given the power and generality of the techniques in this paper, it should not
> be hard to figure out which of these explanations is correct (e.g., to check
> (c) you can just set all the radius errors to zero), or whether there is some
> other explanation.
>
>
>
> Scientific Editor comments:
>
> a) In Fig 1 and similar diagrams, can a scale to the gray squares be provided?
> Can the values of the contours be provided?
>
> b) Related to the referee's point 5b, can you discuss the role of histogram
> binning in the difference between your and previous results? Earlier
> researchers choose arbitrarily chosen bins in mass or period or radius, and are
> subject to bins with few data points. Is your method bin-free? Can your low
> occurrence rate of Earths be approached using inverse-detection-efficiency if
> different binning procedures were chosen?
>
> c) Again related to the referee's 5b, can you clarify the relative importance
> of smoothing vs. binning (eqns 14 vs 12) and MLE vs. Bayesian (i.e. use of a
> prior, eqn 16)?
>
> d) Is the quantity in eqn 26 the same as the eta_\earth quantities in other
> studies. If so, please use established nomenclature; if not, please clarify for
> the reader.
>
> e) The Journal is moving towards an electronic-only format, and increasingly
> encourages authors to provide electronic materials to accompany the text and
> figures. For example, the real and simulated data points in Figs 1-2-5 could be
> provided as `Data behind the Figure'
> (http://aas.org/authors/manuscript-preparation-aj-apj-author-instructions).
