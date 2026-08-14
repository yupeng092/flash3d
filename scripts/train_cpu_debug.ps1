# CPU smoke-test training for the Depth Anything V2 Base integration.
# Run from the project root:
#   .\scripts\train_cpu_debug.ps1
param(
    [string]$DataPath = "data/RealEstate10K",
    [int]$Epochs = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

python train.py +experiment=layered_re10k_cpu_debug `
    "dataset.data_path=$DataPath" `
    "optimiser.num_epochs=$Epochs"
