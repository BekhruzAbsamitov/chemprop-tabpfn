# Archive

Scripts and cluster job files from earlier stages of the project. None of them produces a result, figure or
table of the thesis; they are kept for the record. They were written to run from `src/scripts/` and `cluster/`
and are not maintained here, so paths inside them may no longer resolve.

## scripts/

| File | What it was for |
|---|---|
| `run_active_learning.py` | first active learning loop, replaced by `benchmark_al.py` and `benchmark_extended.py` |
| `evaluate_adaptive.py`, `model_evaluation.py` | early evaluation of trained encoders on the old pool files |
| `sample_efficiency.py` | sample-efficiency study that is not part of the thesis |
| `sanity_overfit.py` | diagnostic: can the encoder overfit a handful of assays |
| `summarise_primary_benchmark.py`, `summarise_extended.py`, `summarise_transfer.py` | summaries of the earlier benchmark runs, replaced by `plot_benchmark.py`, `plot_training.py` and the `appendix_*_tables.py` scripts |
| `plot_curves.py`, `plot_epoch_curves.py`, `plot_lr_comparison.py`, `plot_epoch_lr_comparison.py` | training-curve figures that are not in the thesis (the thesis figures come from `plot_training.py`) |
| `plot_acquisition.py` | illustration of the acquisition functions, removed from the Background chapter |

## cluster/

Job files of the earlier benchmark runs (`run_gpu_benchmark*`, `run_gpu_benchmark_extended*.sub`), which the final
run (`cluster/run_gpu_benchmark_final.sub`) replaced; the first runtime sweep on a different GPU
(`run_gpu_runtime.sub`); the cosine-schedule training run, which the thesis does not report
(`run_gpu_full_sp_10ep_cosine.*`); and the first active learning job (`run_al.*`).
