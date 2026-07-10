"""PyTorch sequence models for per-week WR fantasy-point prediction (research).

Learning sub-project on the ``pytorch-experiments`` branch: build a sequence
model (LSTM first, then a Transformer) that reads each WR's weekly feature
vectors *in order*, and benchmark it head-to-head against the
HistGradientBoosting baseline in ``src/research/wr_weekly_model.py``
(OOF R2 = 0.024, MAE = 9.05).

Honest framing: for ~3k-row tabular regression, gradient-boosted trees usually
win. PyTorch is here for the *learning* + a defensible "classical vs deep on
tabular sequence data" portfolio comparison -- not for guaranteed predictive
lift. "The GBM still wins" is an acceptable, interview-worthy outcome.

Planned module layout (built milestone-by-milestone -- see the full plan in
``docs/research/wr_weekly_torch.md``):

- ``data.py``   -- SequenceDataset + collate_fn (padding / attention masks);
                   reuses the *exact* KFold split from ``wr_weekly_model.py`` so
                   the comparison against the GBM is apples-to-apples.
- ``models.py`` -- WRSequenceLSTM and WRSequenceTransformer (``nn.Module``).
- ``train.py``  -- architecture-agnostic train/eval loop, gradient clipping,
                   masked loss, and ``state_dict`` checkpointing.
- ``run.py``    -- CLI entry point: run KFold, print GBM-comparable OOF R2/MAE.

Run (once built), from the repo root:

    .venv\\Scripts\\python.exe -m src.research.wr_torch.run     # Windows
    .venv/bin/python        -m src.research.wr_torch.run     # macOS / Linux
"""
