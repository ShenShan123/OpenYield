# Yield-estimation integration provenance

This integration is based on `ShenShan123/OpenYield` commit
`578d53d502d6418970447c1f34accfe3d0b6b957`.

The reusable interface, EFIAL, MNIS and structured-result design were ported
and adapted from the Apache-2.0 repository
`IceLab-JCIE/EDA26-Yield-Array-Transfer-Nanlin` at
`88b54f2693b019cf3fcc71a283292140ce94d5a5`.

The ACS and HSCS SRAM adaptation was reviewed at
`d6410f71337707c252b2ad0eb41904350d5952dc`. The additional legacy algorithm
inventory was reviewed at `9fcc49977268318be0aa256876fe9f005f4d4295`.

The credential-bearing legacy launcher was not copied. FUSIS, OPT and BIBD
are exposed only as experimental APIs; FUSIS is not claimed to reproduce the
publication exactly.
