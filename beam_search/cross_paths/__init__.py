"""
cross_paths — beam search over GP-predicted sequence property space.

Named capabilities (public API surface for future al_pipeline integration):
  - beam_search         : beam_search_paths() in beam_search.py
  - model_io            : load_model_bundle() in model_io.py
  - resume              : --resume flag in run_beams_mpi.py
  - extend_no_finished  : --extend_no_finished flag
  - ps_endpoint_append  : append_missing_ps_endpoints.py
  - stagnation          : --stagnation_patience / --stagnation_delta in beam_search.py

Future direction: candidate for absorption into al_pipeline as al_pipeline.beam_search.
The MPI dispatch pattern in run_beams_mpi.py may remain as a standalone cluster entry
point even after that integration.
"""
