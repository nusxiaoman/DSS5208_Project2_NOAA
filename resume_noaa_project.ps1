# resume_noaa_project.ps1
Write-Host "🚀 Resuming NOAA Weather Project..." -ForegroundColor Green

# Set project
gcloud config set project distributed-map-475111-h2
gcloud config set dataproc/region asia-southeast1

# Set environment variables
$env:PROJECT_ID = "distributed-map-475111-h2"
$env:REGION = "asia-southeast1"
$env:BUCKET = "weather-ml-bucket-1760514177"

# Check running jobs
Write-Host "`n📊 Current Dataproc Batches:" -ForegroundColor Cyan
gcloud dataproc batches list --region=asia-southeast1 --limit=5

Write-Host "`n✅ Project resumed! Ready to work." -ForegroundColor Green


# When you return to this project
# .\resume_noaa_project.ps1