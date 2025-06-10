# scripts/rotate-logs.ps1
# Simple log rotation script (placeholder)

$LogFile = "C:\Users\patk0\OneDrive\Desktop\MT5Bot\logs\signals.log"
$MaxBytes = 10485760 # 10 MB
$BackupCount = 7

# Check if log file exists and exceeds max size
if (Test-Path $LogFile) {
    $fileInfo = Get-Item $LogFile
    if ($fileInfo.Length -ge $MaxBytes) {
        Write-Host "Log file $LogFile exceeds max size. Rotating..."

        # Implement actual rotation logic here (e.g., renaming, compressing)
        # For demonstration, just show a message
        Write-Host "Log rotation logic goes here."

        # Example: Delete oldest backup if count exceeds limit
        # Get-ChildItem "C:\Users\patk0\OneDrive\Desktop\MT5Bot\logs\snapshots\signals.log.*" | Sort-Object LastWriteTime | Select-Object -First ($_.Count - $BackupCount) | Remove-Item
    }
}

# PowerShell script for daily log rotation
# This is a placeholder. Implement actual log rotation logic based on your needs.

Write-Host "Rotating logs..."

# Example: Implement your log rotation logic here
# For a simple file rotation, you might use a library in Python or a more complex script here.

Write-Host "Log rotation complete." 