# LIF

中文交互模拟器。支持恒定／正弦电流、参数调节、四联图及脉冲统计；鼠标悬停可查看数值。导出在 exe 同目录的 output 下新建独立目录，保存 trace.csv 和 parameters.json。

## 下载运行

从 [GitHub Releases](https://github.com/LevelDownRefine/neural-network-teaching/releases) 下载 lif-v版本号-windows-x64.zip，解压到可写目录后运行 lif.exe。无需安装 Python、Rust 或 VC++ 运行库。

## 构建

需要 Rust 1.88 或更新版本；Windows 还需 MSVC Build Tools。

```sh
cargo run --release --locked --manifest-path lif/simulator/Cargo.toml
```

在 Windows PowerShell 中生成发布包：

```powershell
./lif/simulator/release.ps1
```

脚本测试界面与核心，静态链接 MSVC 运行时，生成 dist/lif-v版本号-windows-x64.zip 和 SHA256SUMS.txt。源码中的 core 是共享仿真核心，src 是原生界面；两者共用 Cargo workspace 和锁文件。

## 发布

合并到 main 后，确认 Cargo.toml 的版本号，再推送对应标签：

```sh
git switch main
git pull --ff-only
git tag lif-v0.1.0
git push origin lif-v0.1.0
```

CI 全部通过后自动创建 GitHub Release，上传压缩包和校验值；标签必须与 Cargo 版本一致。

PR 和 main 推送运行检查；Rust 检查及 Windows 打包按相关文件变更触发，发布标签强制执行全部检查。同一 PR 的新提交会取消旧构建。

## Python 对照

Notebook 和训练使用 snnTorch。运行 uv sync --locked --extra cpu --extra rust 可通过 maturin 安装 core 的 Python 扩展。

Rust 测试覆盖阈值、重置、不应期、参数更新及导出；pytest 将恒定、正弦和随机输入的结果与 snnTorch 逐元素对照。性能单独测量，不以机器相关的耗时阈值判断 CI 是否通过。

本机 20,000 步实验，Rust Python 接口约 1.2–1.3 ms，原实现约 3.3–3.5 s，仅比较仿真，不含绘图。
