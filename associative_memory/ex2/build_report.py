"""Build the Chinese Markdown report from the saved experiment results."""

import csv
import json

import numpy as np

from .run import CONFIG, OCCLUSIONS, OUTPUT


def read_rows(name):
    with (OUTPUT / name).open(encoding="utf-8", newline="") as stream:
        return [
            {k: v if k == "method" else float(v) for k, v in row.items()}
            for row in csv.DictReader(stream)
        ]


def make_sections(main, repeated, summary):
    sections = [
        (
            "实验设置与指标",
            [
                "实验固定 d=64、m=32、P=32、学习率 0.5。每条线索的坐标独立等概率取 -1/8 或 +1/8，"
                "内容的坐标独立等概率取 -1 或 +1。Hebbian 与 Delta 均从零矩阵开始，每个配对只写入一次。",
                "主实验随机种子为 42。两种规则共享配对、随机写入顺序和全部查询掩码。遮挡比例依次为 0%、25%、50%、75%，"
                "每条记忆在每个比例生成 10 个掩码，分别保留恰好 64、48、32、16 个坐标。遮挡后将查询除以其 L2 范数。",
                "每个方法、每个遮挡比例评估 320 次检索，共比较 10240 个内容比特。比特正确率对这些比特取平均；"
                "MSE 对连续输出 z 与真实内容之差的平方取平均；完整恢复率要求一次检索的 32 位全部正确。"
                "零输出按模型约定判为 +1。",
                "检索只输入查询，不提供真实内容，不更新权重。0% 遮挡的 10 次查询相同，指标等于每条记忆只查询一次的结果，"
                "不能把这 10 次视作独立数据。",
            ],
        ),
        (
            "主实验结果",
            [
                "遮挡从 0% 增至 75% 时，Hebbian 比特正确率从 91.80% 降至 76.19%，Delta 从 95.80% 降至 77.01%；"
                "MSE 分别从 0.3536 升至 0.6795、从 0.3587 升至 0.6983。两种规则都随线索缺失而退化。",
                "四个遮挡比例下，Delta 的正确率分别比 Hebbian 高 4.00、2.50、1.38、0.82 个百分点；"
                "同时 Delta 的 MSE 在四个比例下都更高。曲线说明遮挡增加时两种方法的正确率差距缩小，"
                "也说明比特正确率和连续读出 MSE 不能互相替代。",
                "75% 遮挡下，两种规则的完整恢复率均为 0%。因此，主实验支持“只剩四分之一线索时仍能恢复多数比特”，"
                "不支持“能够完整恢复整条内容”。这里的多数比特高于随机猜测对平衡随机内容的 50% 期望正确率。",
            ],
        ),
        (
            "20 组独立重复实验",
            [
                "为检查主实验趋势，额外使用种子 0 至 19。每个种子重新生成配对、顺序和掩码，"
                "同一种子内两种规则仍共享所有条件。表中给出 20 组独立实验的均值与样本标准差，"
                "误差条表示标准差，不是置信区间。没有根据测试结果调整学习率或选择种子。",
                "0%、25%、50% 遮挡时，Delta 的比特正确率在 20/20 组配对比较中都高于 Hebbian。"
                "75% 遮挡时仅有 12/20 组更高，平均优势只有 0.10 个百分点，不能据此声称 Delta 在严重遮挡时仍稳定占优。",
                "75% 遮挡时，Hebbian 与 Delta 的平均比特正确率分别为 76.44% 和 76.54%，"
                "完整恢复率仅为 0.09375% 和 0.06250%。严重遮挡下，两种规则保留了部分内容信息，完整恢复已接近失败。",
            ],
        ),
        (
            "可视化与样例选择",
            [
                "在查看结果前固定选择主实验的记忆编号 0、1、2，各用第 0 个掩码，展示全部四种遮挡比例。"
                "共展示 12 组查询，每组同时给出真实内容和两种规则的检索内容。样例没有根据成功或失败筛选，"
                "总体结论使用全部记忆和掩码的统计。",
                "残缺线索按 8×8 排列，真实内容和检索内容按 4×8 排列。黑色表示 -1，白色表示 +1，"
                "线索中的灰色表示被遮挡。图中显示线索的符号，实际计算使用归一化后的数值；二维排列仅用于显示，"
                "随机向量没有图像的空间语义。每张检索图下方标注该次查询的比特正确率。",
                "不同遮挡比例独立抽取掩码，保留位置不构成嵌套关系；单个样例的正确率可能非单调。"
                "例如记忆 0 的 50% 遮挡查询优于 25% 遮挡查询，反映的是具体保留坐标的影响。总体趋势由全部查询统计确定。",
            ],
        ),
        (
            "思考与解释",
            [
                "1. 为什么部分线索仍能提取内容？Hebbian 写入后 W=0.5×Σ(v_i k_i^T)。查询第 j 条记忆时，"
                "输出由目标项 0.5×v_j×(k_j^T q) 和其他记忆的串扰项组成。保留 r 个坐标并归一化后，"
                "k_j^T q=√(r/64)。目标内容仍会进入输出，所以残缺线索仍具有检索能力。",
                "2. 为什么遮挡越多错误越多？四种设置的目标相关度依次为 1、0.8660、0.7071、0.5。"
                "对独立随机键，其他线索与单位查询内积的方差为 1/64。Hebbian 的单比特串扰方差期望为"
                "0.5²×31/64=0.1211，并不因遮挡后再次归一化而同比下降。目标信号减弱而串扰仍在，符号更容易翻错。"
                "这一分解针对单遍 Hebbian；Delta 的权重还依赖历史预测误差，不能直接套用同一串扰公式。",
                "3. 为什么 Delta 正确率更高，MSE 却更大？正确率只检查输出符号，MSE 还惩罚输出幅值偏离 ±1。"
                "因为真实比特平方为 1，MSE=1+平均(z²)-2×平均(zv)。20 组完整线索测试中，Hebbian 的平均(z²)"
                "与平均(zv)分别为 0.3786、0.5067，Delta 分别为 0.2687、0.4481，得到 MSE 0.3653 与 0.3726。"
                "Delta 的输出能量较小，但目标对齐项也更小，最终 MSE 更大。符号正确率更高不保证连续值更接近目标。",
                "4. 它恢复的是什么？计算流程为查询 q→矩阵乘法 Wq→符号输出内容。输出维数为 32，"
                "输入线索维数为 64；输出没有反馈到输入，也没有循环迭代。因此这是由残缺线索检索配对内容，"
                "当前系统没有执行输入线索的迭代补全。",
            ],
        ),
        (
            "结论",
            [
                "本实验的明确结论是：残缺线索可以支持部分内容检索；遮挡越重，比特正确率越低、MSE 越大。"
                "Delta 在 0% 至 50% 遮挡下的比特正确率优势在本次 20 组重复实验中一致出现，75% 遮挡下优势不稳定。"
                "两种规则在 75% 遮挡时都只能恢复部分比特，完整恢复率低于 0.1%。这些结论限于本实验的随机独立配对、"
                "单遍写入、固定负载及学习率，不外推到其他设置。",
            ],
        ),
    ]
    # Check all numerical comparisons used by the report against actual results.
    for p in OCCLUSIONS:
        a = [r for r in repeated if r["method"] == "Delta" and r["occlusion"] == p]
        b = [r for r in repeated if r["method"] == "Hebbian" and r["occlusion"] == p]
        wins = sum(x["bit_accuracy"] > y["bit_accuracy"] for x, y in zip(a, b))
        assert wins == (12 if p == 0.75 else 20)
    for method in ("Hebbian", "Delta"):
        series = [r for r in main if r["method"] == method]
        assert all(
            x["bit_accuracy"] > y["bit_accuracy"] for x, y in zip(series, series[1:])
        )
        assert all(x["mse"] < y["mse"] for x, y in zip(series, series[1:]))
    for row in main + repeated:
        np.testing.assert_allclose(
            row["mse"], 1 + row["output_power"] - 2 * row["target_alignment"]
        )
    return sections


def markdown_report(sections, main, summary, raw):
    lines = [
        "# 实验二：线索不完整还能想起来吗",
        "",
        "## 作业要求与提交结果对应",
        "",
        "| 作业要求 | 本报告中的结果 | 直接结论 |",
        "|---|---|---|",
        "| 遮挡比例与正确率、MSE 的关系 | 主实验指标表、主实验曲线及 20 组重复实验曲线 | 遮挡越重，正确率越低，MSE 越高 |",
        "| 至少三组‘残缺线索—真实内容—检索内容’并说明选取方式 | 记忆 0、1、2 的独立 PNG 图及下方逐例对照表 | 图示是固定样例，整体趋势由全部查询统计确定 |",
        "| 解释恢复能力与局限 | 思考与解释、结论 | 模型由残缺线索检索配对内容，没有迭代补全输入线索 |",
        "",
        "以下依次给出实验方法、数值结果、三组样例以及对上述结论的解释。",
        "",
    ]
    for index, (title, paragraphs) in enumerate(sections):
        lines += [f"## {title}", "", *[p + "\n" for p in paragraphs]]
        if index == 1:
            lines += [
                "| 方法 | 遮挡比例 | 比特正确率 | MSE | 完整恢复率 |",
                "|---|---:|---:|---:|---:|",
            ]
            lines += [
                f"| {r['method']} | {r['occlusion']:.0%} | {r['bit_accuracy']:.2%} | "
                f"{r['mse']:.4f} | {r['exact_recall_rate']:.2%} |"
                for r in main
            ]
            lines += ["", "![主实验曲线](results/main_curves.png)", ""]
        if index == 2:
            lines += [
                "| 方法 | 遮挡比例 | 正确率均值 ± 标准差 | MSE 均值 ± 标准差 |",
                "|---|---:|---:|---:|",
            ]
            lines += [
                f"| {r['method']} | {r['occlusion']:.0%} | "
                f"{r['bit_accuracy_mean']:.2%} ± {r['bit_accuracy_sd']:.2%} | "
                f"{r['mse_mean']:.4f} ± {r['mse_sd']:.4f} |"
                for r in summary
            ]
            lines += [
                "",
                "![重复实验曲线，误差条为跨种子标准差](results/repeat_curves.png)",
                "",
            ]
        if index == 3:
            lines += [
                "| 记忆编号 | 遮挡比例 | 保留坐标 | Hebbian 正确比特 | Delta 正确比特 |",
                "|---:|---:|---:|---:|---:|",
            ]
            for memory in CONFIG["example_memory_indices"]:
                for level, p in enumerate(OCCLUSIONS):
                    counts = [
                        int(
                            np.sum(
                                np.where(raw["z"][method, level, memory, 0] >= 0, 1, -1)
                                == raw["values"][memory]
                            )
                        )
                        for method in (0, 1)
                    ]
                    lines += [
                        f"| {memory} | {p:.0%} | {int(raw['masks'][level, memory, 0].sum())}/64 | "
                        f"{counts[0]}/32 | {counts[1]}/32 |"
                    ]
            lines += [""]
            for memory in CONFIG["example_memory_indices"]:
                lines += [
                    f"### 记忆 {memory}",
                    "",
                    f"![记忆 {memory} 的四种遮挡查询](results/example_{memory}.png)",
                    "",
                ]
            lines += [
                "这三条记忆的固定掩码展示了不同的局部结果。例如记忆 0 在 25% 遮挡时，"
                "Hebbian 正确 28/32 位，在 50% 遮挡时正确 30/32 位。"
                "因为各遮挡比例独立抽取保留位置，单个样例可以非单调；"
                "是否整体退化应以主实验表格的 320 次检索为准。",
                "",
            ]
    lines += [
        "## 复现与文件",
        "",
        "在仓库根目录激活含 NumPy、Matplotlib、PyTorch 的环境后运行：",
        "",
        "```powershell",
        "python -m associative_memory.ex2.run",
        "python -m associative_memory.ex2.build_report",
        "```",
        "",
        "主实验明细为 results/main_metrics.csv，20 组重复明细为 repeat_metrics.csv，"
        "汇总为 repeat_summary.csv。main_raw.npz 保存主实验的配对、写入顺序、掩码、归一化查询、"
        "两种权重和全部连续输出，数组轴定义见 config.json。",
        "",
        "本实验复用 associative_memory/data.py、models/model_memory.py 和 train.py。"
        "实现依据项目根目录 Homework3_Plasticity_Based_Associative_Memory.pdf 第 1—2 页。",
        "",
        "自动检查包括掩码保留数量、查询单位范数、检索不修改权重、输出有限性、"
        "Hebbian 闭式解，以及报告中的指标分解和趋势。",
    ]
    (OUTPUT.parent / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    saved_config = json.loads((OUTPUT / "config.json").read_text(encoding="utf-8"))
    assert saved_config["main_seed"] == 42 and saved_config["repeat_seeds"] == list(
        range(20)
    )
    main_rows = read_rows("main_metrics.csv")
    repeats = read_rows("repeat_metrics.csv")
    summary = read_rows("repeat_summary.csv")
    with np.load(OUTPUT / "main_raw.npz") as stored:
        raw = {key: stored[key] for key in ("z", "values", "masks")}
    sections = make_sections(main_rows, repeats, summary)
    markdown_report(sections, main_rows, summary, raw)
    print("Report generated: associative_memory/ex2/report.md")


if __name__ == "__main__":
    main()
