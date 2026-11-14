$ErrorActionPreference = 'Stop'
$manifest = Join-Path $PSScriptRoot 'Cargo.toml'
$target = 'x86_64-pc-windows-msvc'

$metadata = & cargo metadata --locked --no-deps --format-version 1 --manifest-path $manifest
if ($LASTEXITCODE -ne 0) { throw '读取 Cargo 元数据失败' }
$metadata = $metadata | ConvertFrom-Json
$version = ($metadata.packages | Where-Object { $_.name -eq 'lif' }).version
if ($env:GITHUB_REF -like 'refs/tags/lif-v*' -and $env:GITHUB_REF_NAME -cne "lif-v$version") {
    throw "发布标签必须与 Cargo 版本一致：lif-v$version"
}

# 静态链接 MSVC 运行时，分发时无需额外安装 VC++ Redistributable。
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS = '-C target-feature=+crt-static'
& cargo test --release --locked --workspace --target $target --manifest-path $manifest
if ($LASTEXITCODE -ne 0) { throw 'Rust 测试失败' }
& cargo build --release --locked --target $target --manifest-path $manifest
if ($LASTEXITCODE -ne 0) { throw '构建失败' }

$dist = Join-Path $PSScriptRoot 'dist'
New-Item -ItemType Directory -Path $dist -Force | Out-Null
$exe = Join-Path $metadata.target_directory "$target/release/lif.exe"
$archiveName = "lif-v$version-windows-x64.zip"
$archive = Join-Path $dist $archiveName
Compress-Archive -LiteralPath $exe, (Join-Path $PSScriptRoot 'README.md') -DestinationPath $archive -Force
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $archiveName" | Set-Content -LiteralPath (Join-Path $dist 'SHA256SUMS.txt') -Encoding ascii
Write-Output "已生成 $archive"
