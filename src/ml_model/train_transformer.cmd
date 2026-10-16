@echo off
REM ===================================================================
REM  Transformer baseline: the third method numbers #2 and #3 need.
REM
REM  Four runs, all through train_clean.py -- the SAME training loop,
REM  split, loss and optimiser the GRU used, so a difference in the
REM  result is a difference in architecture and not in training.
REM
REM    matched (d_model 32, 1 layer, ff x4) = 13,539 params at each of
REM    the three horizons, against the GRU's 14,019. Then one larger
REM    run (d_model 64, 2 layers) = 101,635 params at 750 ms, so
REM    "it only lost because it was small" can be answered with a
REM    measurement instead of an opinion.
REM
REM  Checkpoints land in clean\tfm{d_model}_h{horizon}_s0.pt, beside
REM  the clean GRU ones, and nothing is overwritten.
REM
REM  Run from src\ml_model. Expect roughly an hour.
REM ===================================================================

if not exist clean mkdir clean

for %%F in (parsed_h4.npz parsed_h8.npz parsed_data.npz) do (
  echo.
  echo ============================================================
  echo  %%F  --  transformer, parameter-matched
  echo ============================================================
  py train_clean.py --input %%F --arch transformer --d-model 32 ^
     --layers 1 --ff-mult 4 --epochs 12 --seed 0 ^
     > clean\log_tfm32_%%~nF.txt 2>&1
  findstr /C:"parameters" /C:"best epoch" /C:"dead reckoning" /C:"Transformer " /C:"beats dead reckoning" /C:"saved" clean\log_tfm32_%%~nF.txt
)

echo.
echo ============================================================
echo  parsed_data.npz  --  transformer, 7.25x the GRU's parameters
echo ============================================================
py train_clean.py --input parsed_data.npz --arch transformer --d-model 64 ^
   --layers 2 --ff-mult 4 --nhead 4 --epochs 12 --seed 0 ^
   > clean\log_tfm64_parsed_data.txt 2>&1
findstr /C:"parameters" /C:"best epoch" /C:"dead reckoning" /C:"Transformer " /C:"beats dead reckoning" /C:"saved" clean\log_tfm64_parsed_data.txt

echo.
echo All four runs finished. Logs in clean\log_tfm*.txt
echo.
echo Compare against clean\log_parsed_*.txt, which hold the GRU runs
echo on the same test games.
