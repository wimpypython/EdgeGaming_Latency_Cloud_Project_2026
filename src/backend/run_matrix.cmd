@echo off
REM ===================================================================
REM  GRU concealment: both rung policies x three loss seeds.
REM
REM  Every run is reproducible: --gap-time seq removes wall-clock
REM  jitter, and --loss-seed fixes the loss pattern. The same seed
REM  under both policies therefore scores the SAME gaps, so the two
REM  policies are compared on identical traffic and the difference
REM  between them is the policy, not the sample.
REM
REM  Three seeds because one run is one sample. The spread across
REM  seeds is the uncertainty on the result and belongs in the report
REM  next to it.
REM
REM  Run from src\backend. Takes about 32 minutes.
REM ===================================================================

if not exist ..\..\results\gru_matrix mkdir ..\..\results\gru_matrix

for %%P in (cover nearest) do (
  for %%S in (1 2 3) do (
    echo [%%P seed %%S] starting
    start /b cmd /c "py server_sync.py --rate 16 --conceal gru --model-dir ..\ml_model --http 8123 --seconds 330 --quiet --gap-time seq --rung-policy %%P > ..\..\results\gru_matrix\%%P_s%%S.txt 2>&1"
    timeout /t 4 /nobreak >nul
    py replay_client.py --npz ..\ml_model\parsed_data.npz --model-dir ..\ml_model --players 10 --seconds 300 --loss --loss-seed %%S --quiet > ..\..\results\gru_matrix\%%P_s%%S_client.txt 2>&1
    timeout /t 12 /nobreak >nul
    echo [%%P seed %%S] done
  )
)

echo.
echo All six runs finished. Results in ..\..\results\gru_matrix\
echo.
for %%F in (..\..\results\gru_matrix\*_s?.txt) do (
  echo === %%~nF
  findstr /C:"The GRU beats" /C:"outage events" /C:"  GRU " /C:"(DR, GRU gaps)" "%%F"
)
