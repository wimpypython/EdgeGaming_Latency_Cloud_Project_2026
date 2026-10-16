@echo off
REM ===================================================================
REM  Retrain all three rungs with a clean validation split.
REM
REM  Each run reads its horizon out of the npz and writes
REM  clean\gru_h{horizon}_s0.pt. The existing checkpoints beside this
REM  script are not touched, so the biased and unbiased numbers can be
REM  compared rather than one replacing the other.
REM
REM  parsed_h1.npz is deliberately NOT retrained: its horizon is 62.5 ms,
REM  below the 125 ms floor under which the live hook does not consult a
REM  model at all, because dead reckoning's error there is a fraction of
REM  a unit.
REM
REM  Run from src\ml_model.
REM ===================================================================

if not exist clean mkdir clean

for %%F in (parsed_h4.npz parsed_h8.npz parsed_data.npz) do (
  echo.
  echo ============================================================
  echo  %%F
  echo ============================================================
  py train_clean.py --input %%F --epochs 12 --seed 0 > clean\log_%%~nF.txt 2>&1
  findstr /C:"Three-way split" /C:"val  games" /C:"test games" /C:"best epoch" /C:"dead reckoning" /C:"GRU " /C:"beats dead reckoning" /C:"saved" clean\log_%%~nF.txt
)

echo.
echo All three retrained. Full logs in clean\log_*.txt
echo Ladder check:
py ..\backend\gru_conceal.py --dir clean
