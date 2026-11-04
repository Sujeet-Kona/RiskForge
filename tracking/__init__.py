"""Offline experiment and model-lifecycle tracking for RiskForge.

This package is the only place in the repository that talks to MLflow, and it
only ever runs offline. It records what produced a model — parameters, library
versions, dataset identity, split sizes, evaluation metrics, the exact feature
contract and the fitted pipeline — so a trained model can be traced back to the
data and code that made it.

It is deliberately *not* part of inference. The FastAPI scoring path loads the
Phase 6 serving artifact from disk and never imports this package, so a
prediction cannot depend on a tracking server being reachable.

Submodules are imported explicitly rather than re-exported here, because
:mod:`tracking.run` pulls in MLflow and the model layers while
:mod:`tracking.config` needs nothing but the standard library.
"""
