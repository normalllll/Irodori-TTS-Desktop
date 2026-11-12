# Dot-source this file before launching this source checkout; no fixed root name.
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $PSScriptRoot '.cache\torch'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
