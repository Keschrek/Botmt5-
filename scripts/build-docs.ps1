# scripts/build-docs.ps1
# MkDocs build script with error exit code

Write-Host "Building documentation with MkDocs..."

try {
  # Run the mkdocs build command
  mkdocs build --clean

  # Check the exit code of the last command
  if ($LASTEXITCODE -ne 0) {
    Write-Error "MkDocs build failed with exit code: $LASTEXITCODE"
    exit $LASTEXITCODE
  } else {
    Write-Host "MkDocs build completed successfully."
  }
} catch {
  Write-Error "An error occurred during MkDocs build: $_"
  exit 1
} 