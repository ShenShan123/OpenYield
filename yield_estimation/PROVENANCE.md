# Yield-estimation integration provenance

This integration is based on `ShenShan123/OpenYield` commit
`578d53d502d6418970447c1f34accfe3d0b6b957`.

The reusable interface, EFIAL, MNIS and structured-result design were ported
and adapted from the Apache-2.0 repository
`IceLab-JCIE/EDA26-Yield-Array-Transfer-Nanlin` at
`88b54f2693b019cf3fcc71a283292140ce94d5a5`.

The CrossTopo MC/MNIS/AIS flows and current repository snapshot were reviewed
at `6eff717edc856504bbb05d1e1bb573e9c1829d39`. The ACS/HSCS SRAM adaptation
was introduced at `512b4ca`, and the explicit ACS `original`/`improved` split
was reviewed at `d6410f71337707c252b2ad0eb41904350d5952dc`. The additional
legacy algorithm inventory was reviewed at
`9fcc49977268318be0aa256876fe9f005f4d4295`.

The normalized MC/MNIS/AIS/ACS/HSCS classes preserve the donor proposal,
clustering, and adaptive-update rules, but replace donor-side simulator path
changes, deletion, global failure caches, and FOM stopping with the shared
`SimulationRunner` and strict charged budget. ACS defaults to `original`.

EFIAL, FUSIS, and OPT use the method-specific donor implementations at
`88b54f2693b019cf3fcc71a283292140ce94d5a5`: iterative density-weighted GMM,
surrogate-guided MCMC correction, and failure-trained CNF mixture,
respectively. Their normalized classes also use `SimulationRunner` and strict
budgets. FUSIS provides an RBF-SVM/NumPy surrogate fallback, and OPT provides a
diagonal-Gaussian fallback when the optional Torch/`nflows` stack is absent;
result metadata records the backend so fallback results cannot be presented as
the donor's learned backend. The FUSIS verification stage uses the normalized
importance correction `E_f[p_hat] * I/p_hat` for its MCMC target density rather
than treating the raw conditional hit rate as the correction factor.

The credential-bearing legacy launcher was not copied. FUSIS, OPT and BIBD
remain experimental APIs.
