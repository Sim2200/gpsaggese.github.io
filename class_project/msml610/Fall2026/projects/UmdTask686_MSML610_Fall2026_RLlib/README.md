# RLlib Optimal Trade Execution

Train RLlib agents to liquidate an order of SPY shares within one trading day,
minimizing the implementation shortfall (bps) versus the arrival price, and
compare them with TWAP, an immediate full sale, and the Almgren-Chriss
schedule.

- Issue: [#686](https://github.com/gpsaggese/gpsaggese.github.io/issues/686)
- Spec:
  [RLlib_Project_Description.md](../../../../project_descriptions/MSML610/RLlib_Project_Description.md),
  Project 1

This is a running doc: status, results, and decisions are updated with each
milestone, newest first.

## Status

| Milestone | Content | Status |
|---|---|---|
| M1 | Container, environment, SPY calibration, TWAP and sell-all baselines | Done |
| M2 | `rllib.API.ipynb`: RLlib API on CartPole | Not started |
| M3 | `rllib.example.ipynb`: DQN and PPO agents, evaluation, plots | Not started |
| Bonus | PPO vs DQN, Almgren-Chriss variance penalty and schedule | Not started |

## Results

### M1: Calibration and Baselines

Setup:

- Order: sell 100,000 SPY shares in 20 steps within one trading day
- Data: SPY 1-minute bars, regular hours, consolidated feed (Alpaca),
  2016-2024
- Volatility: std of the open-to-close log return
- Impact: coefficients from Almgren, Thum, Hauptmann, and Li (2005), scaled by
  SPY volatility, linear in the traded fraction of daily volume
- Baselines: 1,000 episodes each, same price paths for both

| | 2016-2019 (calibration) | 2020-2024 (test) |
|---|---|---|
| Trading days | 1006 | 1258 |
| Open-to-close daily vol (%) | 0.66 | 0.94 |
| Avg daily volume (M shares) | 72.0 | 69.7 |
| Order as % of ADV | 0.139 | 0.143 |
| gamma (permanent) | 0.00208 | 0.00295 |
| eta (temporary) | 0.0188 | 0.0267 |
| TWAP shortfall, mean ± std (bps) | -0.06 ± 36.35 | -0.08 ± 51.51 |
| Sell-all shortfall, mean ± std (bps) | 0.26 ± 0.00 | 0.38 ± 0.00 |

Findings:

- 100,000 shares is 0.14% of SPY's daily volume, so the impact cost is below
  0.4 bps
- TWAP shortfall is driven by price noise (~36-52 bps std), not by impact
- The order size needs to be a larger fraction of daily volume for the agent
  to have a cost to optimize

## Decisions

- 2026-10-22: SPY 1-minute bars instead of daily data, following GP's guidance
  that higher resolution is better
- 2026-10-22: Alpaca as the data source, since its free tier has 1-minute
  history and yfinance does not
  - Alpaca history starts in 2016, so the calibration period is 2016-2019
    instead of 2015-2019
  - Check against Yahoo daily: all 2,264 trading days match, close within
    0.17 bps (median), regular-hours volume ~85% of the Yahoo daily volume

## Pending

- WRDS access to NYSE TAQ (Trade and Quote) requested on 2026-10-22, waiting
  for UMD approval. If approved:
  - Calibrate on 2015-2019, as in the spec
  - Measure the bid-ask spread cost from the quotes
  - Fit the temporary and permanent impact from SPY's own signed trades,
    instead of the Almgren, Thum, Hauptmann, and Li (2005) coefficients

## Files

- `rllib_utils.py`: `ExecutionEnv` Gymnasium environment, TWAP and sell-all
  baselines
- `data_utils.py`: SPY and VIX download, calibration, M1 table
- `test/`: unit tests for both files
- `rllib.API.ipynb`, `rllib.example.ipynb`: notebooks (M2, M3)
- `Dockerfile`, `requirements.txt`, `docker_*.sh`: container, see
  `docker_scripts.README.md`

## How to Run

- Build the container and start Jupyter
  ```bash
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```

- Download the data into `data/` (not committed) with a free Alpaca account,
  passing the keys as `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY`
  ```bash
  > docker run --rm --env-file <keys.env> -v $(git rev-parse --show-toplevel):/git_root \
      -w /git_root -e PYTHONPATH=/git_root:/git_root/helpers_root \
      gpsaggese/umd_project_rllib python -c \
      'import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.data_utils as d; d.download_vix(); d.download_spy(2016, 2024)'
  ```

- Reproduce the M1 table
  ```python
  import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.data_utils as d
  print(d.compute_m1_table({"2016-2019": (2016, 2019), "2020-2024": (2020, 2024)}))
  ```

- Run the unit tests
  ```bash
  > pytest test/test_rllib_utils.py test/test_data_utils.py
  ```
