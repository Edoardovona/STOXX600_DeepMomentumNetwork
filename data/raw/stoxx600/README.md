# STOXX 600 Raw Data

This folder contains the single Bloomberg export used as the sole raw data
source for the project: **`SXXR.xlsx`**.

## File

- `data/raw/stoxx600/SXXR.xlsx` — Bloomberg terminal pull, 2006-01-01 to
  2026-05-06, with four sheets:
  - **`stocks`** — wide price matrix, dates × tickers. Column headers include
    the Bloomberg " Equity" suffix (e.g. `"AD NA Equity"`), stripped on load.
    Empty cells mean either the local exchange was closed that day, or the
    ticker had no data that day; both become `NaN` and are disambiguated by
    the universe mask below.
  - **`year_by_year`** — no header row; one column per calendar year
    (`START_YEAR` onward), each listing that year's STOXX 600 constituents.
    This is the point-in-time universe used to mask survivorship bias out of
    the `stocks` sheet.
  - **`unique_tickers`** — union of all tickers ever in the index over the
    period (reference only; not required by the pipeline).
  - **`benchmark`** — SXXR Index daily closing levels (cap-weighted STOXX 600
    benchmark).

## How it's used

`notebooks/01_data_loading.ipynb` is the only place this file is read. It:

1. Loads `stocks`, strips the " Equity" suffix, coerces to numeric, restricts
   to `[start_year, end_year]` from `configs/default.yaml`.
2. Loads `year_by_year` and builds a point-in-time universe mask, zeroing out
   `(date, ticker)` cells where the ticker wasn't a constituent that year.
3. Forward-fills short gaps (capped at 5 trading days) and flags stale prices,
   so downstream returns/volatility aren't corrupted by filled or repeated
   values.
4. Loads `benchmark` for the SXXR series, and builds a synthetic equal-weighted
   (EW) index from the masked `stocks` panel.
5. Writes the cleaned long-format panel and benchmark series to
   `data/processed/stoxx600/`.

## Notes

- This file is expected to be kept unchanged; all cleaning, masking, and
  feature engineering happens downstream in `notebooks/01_data_loading.ipynb`.
- The file is gitignored — you need your own Bloomberg export
  (`year_by_year`, `stocks`, `unique_tickers`, `benchmark` sheets, same
  column/date conventions) to reproduce the pipeline from raw data.