param(
    [Parameter(Mandatory = $true)][string]$ComposeFile,
    [Parameter(Mandatory = $true)][string]$BackupPath,
    [string]$RestoreDatabase = 'razortrust_restore_drill',
    [switch]$KeepRestoreDatabase
)

$ErrorActionPreference = 'Stop'
$arguments = @(
    'scripts/backup_restore_drill.py',
    '--compose-file', $ComposeFile,
    '--backup-path', $BackupPath,
    '--restore-database', $RestoreDatabase
)
if ($KeepRestoreDatabase) {
    $arguments += '--keep-restore-database'
}

python @arguments
if ($LASTEXITCODE -ne 0) {
    throw ('backup/restore drill failed with exit code ' + $LASTEXITCODE)
}
