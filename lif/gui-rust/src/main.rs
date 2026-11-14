#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use eframe::egui::{self, Color32, FontId, RichText, Stroke};
use egui_plot::{HLine, Line, LineStyle, Plot, PlotBounds, VLine};
use lif_native::experiment::{Experiment, Parameters, DT, REST};

const BG: Color32 = Color32::from_rgb(12, 16, 23);
const PANEL: Color32 = Color32::from_rgb(20, 26, 36);
const BORDER: Color32 = Color32::from_rgb(43, 54, 70);
const CYAN: Color32 = Color32::from_rgb(70, 215, 229);
const AMBER: Color32 = Color32::from_rgb(255, 184, 81);
const GREEN: Color32 = Color32::from_rgb(120, 230, 171);
const MUTED: Color32 = Color32::from_rgb(145, 160, 181);

struct Lab {
    parameters: Parameters,
    result_parameters: Parameters,
    result: Experiment,
    elapsed_ms: f64,
    view_ms: f64,
    status: String,
}

impl Lab {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        load_chinese_font(&cc.egui_ctx);
        cc.egui_ctx.set_theme(egui::ThemePreference::Dark);
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = BG;
        visuals.window_fill = PANEL;
        visuals.extreme_bg_color = BG;
        visuals.selection.bg_fill = CYAN.gamma_multiply(0.3);
        visuals.selection.stroke.color = CYAN;
        cc.egui_ctx.set_visuals(visuals);
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(12.0, 10.0);
        style.spacing.slider_width = 150.0;
        style
            .text_styles
            .insert(egui::TextStyle::Body, FontId::proportional(14.0));
        style
            .text_styles
            .insert(egui::TextStyle::Small, FontId::proportional(11.0));
        cc.egui_ctx.set_style(style);
        let parameters = Parameters::default();
        let begin = Instant::now();
        let result = Experiment::run(&parameters).expect("valid initial parameters");
        Self {
            result_parameters: parameters.clone(),
            parameters,
            result,
            elapsed_ms: begin.elapsed().as_secs_f64() * 1000.0,
            view_ms: 200.0,
            status: "已就绪".into(),
        }
    }

    fn recompute(&mut self) {
        let begin = Instant::now();
        match Experiment::run(&self.parameters) {
            Ok(result) => {
                self.result = result;
                self.result_parameters = self.parameters.clone();
                self.elapsed_ms = begin.elapsed().as_secs_f64() * 1000.0;
                self.status = "实时更新".into();
            }
            Err(message) => {
                self.parameters = self.result_parameters.clone();
                self.status = format!("参数无效：{message}");
            }
        }
    }

    fn export_to(&self, root: &Path) -> io::Result<PathBuf> {
        // 新目录一次性容纳数据和参数，绝不覆盖已有导出。
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let folder = root.join(format!("run-{stamp}-{}", std::process::id()));
        fs::create_dir_all(root)?;
        fs::create_dir(&folder)?;
        let mut csv = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(folder.join("trace.csv"))?,
        );
        writeln!(
            csv,
            "input_time_ms,time_ms,current_nA,voltage_mV,spike,rate_Hz"
        )?;
        for k in 0..self.result.spikes.len() {
            writeln!(
                csv,
                "{},{},{},{},{},{}",
                k as f64 * DT,
                (k + 1) as f64 * DT,
                self.result.current[k],
                self.result.voltage[k],
                self.result.spikes[k],
                self.result.rate[k]
            )?;
        }
        csv.flush()?;
        let metadata = serde_json::to_string_pretty(&serde_json::json!({
            "parameters": self.result_parameters,
            "dt_ms": DT,
            "rest_mV": REST,
        }))
        .map_err(io::Error::other)?;
        let mut json = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(folder.join("parameters.json"))?;
        writeln!(json, "{metadata}")?;
        Ok(folder)
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(14.0);
        ui.label(RichText::new("实验设置").color(MUTED).size(11.0).strong());
        ui.add_space(8.0);
        ui.heading("单神经元实验");
        ui.label(RichText::new("调节参数，实时观察响应").color(MUTED));
        ui.add_space(16.0);
        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= ui
                .selectable_value(&mut self.parameters.sinusoidal, false, "恒定电流")
                .changed();
            changed |= ui
                .selectable_value(&mut self.parameters.sinusoidal, true, "正弦电流")
                .changed();
        });
        ui.add_space(14.0);
        ui.label(RichText::new("输入电流").color(CYAN).small().strong());
        changed |= control(
            ui,
            "基础电流",
            "nA",
            &mut self.parameters.current,
            0.0..=4.0,
            0.1,
        );
        let phase_limit = (std::f64::consts::PI * 100.0).floor() / 100.0;
        ui.add_enabled_ui(self.parameters.sinusoidal, |ui| {
            changed |= control(
                ui,
                "振幅",
                "nA",
                &mut self.parameters.amplitude,
                0.0..=3.0,
                0.1,
            );
            changed |= control(
                ui,
                "频率",
                "Hz",
                &mut self.parameters.frequency,
                0.0..=50.0,
                0.5,
            );
            changed |= control(
                ui,
                "相位",
                "rad",
                &mut self.parameters.phase,
                -phase_limit..=phase_limit,
                0.01,
            );
        });
        ui.add_space(12.0);
        ui.separator();
        ui.label(RichText::new("神经元").color(AMBER).small().strong());
        changed |= control(
            ui,
            "发放阈值",
            "mV",
            &mut self.parameters.threshold,
            -60.0..=-40.0,
            1.0,
        );
        changed |= control(
            ui,
            "膜时间常数",
            "ms",
            &mut self.parameters.tau,
            5.0..=50.0,
            1.0,
        );
        changed |= control(
            ui,
            "膜电阻",
            "MΩ",
            &mut self.parameters.resistance,
            1.0..=30.0,
            1.0,
        );
        changed |= control(
            ui,
            "不应期",
            "ms",
            &mut self.parameters.refractory,
            0.0..=10.0,
            0.1,
        );
        ui.add_space(10.0);
        ui.separator();
        ui.label(RichText::new("仿真时间").color(MUTED).small().strong());
        changed |= control(
            ui,
            "仿真时长",
            "ms",
            &mut self.parameters.duration,
            100.0..=5000.0,
            DT,
        );
        ui.horizontal(|ui| {
            ui.label("显示范围");
            for value in [200.0, 500.0, 0.0] {
                let label = if value == 0.0 {
                    "全部".into()
                } else {
                    format!("前 {value:.0} ms")
                };
                ui.selectable_value(&mut self.view_ms, value, label);
            }
        });
        ui.add_space(12.0);
        if ui.button("↺  重置参数").clicked() {
            self.parameters = Parameters::default();
            self.view_ms = 200.0;
            changed = true;
        }
        if changed {
            self.recompute();
        }
        ui.add_space(8.0);
        ui.label(
            RichText::new("静息电位 −65 mV  /  时间步 0.1 ms")
                .color(MUTED)
                .small(),
        );
        ui.label(
            RichText::new("Euler 积分 · 发放后立即回到静息电位")
                .color(MUTED)
                .small(),
        );
        ui.add_space(12.0);
        if ui.button("导出数据与参数").clicked() {
            self.status = match self.export_to(Path::new("lif/output")) {
                Ok(path) => format!("已保存至 {}", path.display()),
                Err(error) => format!("导出失败：{error}"),
            };
        }
        ui.label(RichText::new(&self.status).color(MUTED).small());
    }

    fn chart(&self, ui: &mut egui::Ui, signal: usize, height: f32) {
        let span = if self.view_ms == 0.0 {
            self.parameters.duration
        } else {
            self.view_ms.min(self.parameters.duration)
        };
        let count = ((span / DT).round() as usize).min(self.result.spikes.len());
        let (title, unit, color) = match signal {
            0 => ("01   输入电流", "nA", CYAN),
            1 => ("02   膜电位", "mV", AMBER),
            2 => ("03   脉冲序列", "脉冲", GREEN),
            _ => ("04   发放率", "Hz", Color32::from_rgb(171, 151, 255)),
        };
        egui::Frame::new()
            .fill(PANEL)
            .stroke(Stroke::new(1.0_f32, BORDER))
            .corner_radius(10)
            .inner_margin(egui::Margin::same(12))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(title).size(11.0).strong().color(color));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(if signal == 3 {
                                "100 ms 滑动窗口"
                            } else {
                                unit
                            })
                            .color(MUTED)
                            .size(11.0),
                        );
                    });
                });
                let points: Vec<[f64; 2]> = (0..count)
                    .map(|k| {
                        let time = if signal == 0 {
                            k as f64 * DT
                        } else {
                            (k + 1) as f64 * DT
                        };
                        let value = match signal {
                            0 => self.result.current[k] as f64,
                            1 => self.result.voltage[k] as f64,
                            2 => 0.5,
                            _ => self.result.rate[k],
                        };
                        [time, value]
                    })
                    .collect();
                let (min, max) = if signal == 2 {
                    (0.0, 1.0)
                } else {
                    let mut min = points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
                    let mut max = points
                        .iter()
                        .map(|p| p[1])
                        .fold(f64::NEG_INFINITY, f64::max);
                    if signal == 1 {
                        min = min.min(REST);
                        max = max.max(self.parameters.threshold);
                    }
                    if signal == 3 {
                        min = 0.0;
                    }
                    let pad = ((max - min) * 0.12).max(if signal == 1 { 1.0 } else { 0.1 });
                    (min - pad, max + pad)
                };
                Plot::new(signal)
                    .height(height)
                    .show_background(false)
                    .show_axes([signal == 3, signal != 2])
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .show(ui, |plot| {
                        plot.set_plot_bounds(PlotBounds::from_min_max([0.0, min], [span, max]));
                        if signal == 2 {
                            for (k, &spike) in self.result.spikes.iter().enumerate().take(count) {
                                if spike != 0 {
                                    plot.vline(
                                        VLine::new("脉冲", (k + 1) as f64 * DT)
                                            .color(color)
                                            .width(1.6_f32),
                                    );
                                }
                            }
                        } else {
                            plot.line(Line::new(title, points).color(color).width(1.8_f32));
                        }
                        if signal == 1 {
                            plot.hline(
                                HLine::new("发放阈值", self.parameters.threshold)
                                    .color(Color32::from_rgb(225, 111, 107))
                                    .style(LineStyle::dashed_dense()),
                            );
                            plot.hline(
                                HLine::new("静息电位", REST)
                                    .color(MUTED)
                                    .style(LineStyle::dashed_dense()),
                            );
                        }
                    });
            });
    }
}

impl Lab {
    fn render(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("header")
            .exact_height(76.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(egui::Margin::symmetric(24, 16)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("LIF").size(30.0).strong().color(CYAN));
                    ui.add_space(20.0);
                    ui.label(RichText::new("观察一个神经元的实时响应").color(MUTED));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new("●  实时").color(GREEN).strong());
                    });
                });
            });
        egui::SidePanel::left("parameters")
            .exact_width(306.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(egui::Margin::same(20)),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(BG)
                    .inner_margin(egui::Margin::same(20)),
            )
            .show(ctx, |ui| {
                let count = self.result.count();
                ui.columns(4, |cols| {
                    metric(
                        &mut cols[0],
                        "脉冲总数",
                        format!("{count}"),
                        "统计覆盖完整仿真时长",
                        CYAN,
                    );
                    metric(
                        &mut cols[1],
                        "平均发放率",
                        format!(
                            "{:.1} Hz",
                            count as f64 / (self.parameters.duration / 1000.0)
                        ),
                        "由实际脉冲计算",
                        AMBER,
                    );
                    metric(
                        &mut cols[2],
                        "首个脉冲",
                        self.result
                            .first_spike()
                            .map(|x| format!("{x:.1} ms"))
                            .unwrap_or("—".into()),
                        "从仿真开始计时",
                        GREEN,
                    );
                    metric(
                        &mut cols[3],
                        "仿真耗时",
                        format!("{:.3} ms", self.elapsed_ms),
                        "FP32 前向计算",
                        Color32::from_rgb(171, 151, 255),
                    );
                });
                ui.add_space(12.0);
                let remaining = (ui.available_height() - 250.0).max(200.0);
                self.chart(ui, 0, remaining * 0.22);
                self.chart(ui, 1, remaining * 0.34);
                self.chart(ui, 2, remaining * 0.15);
                self.chart(ui, 3, remaining * 0.29);
                ui.label(
                    RichText::new("时间 / ms     ·     鼠标悬停可查看数值")
                        .color(MUTED)
                        .small(),
                );
            });
    }
}

impl eframe::App for Lab {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.render(ctx);
    }
}

fn control(
    ui: &mut egui::Ui,
    label: &str,
    unit: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    step: f64,
) -> bool {
    ui.label(RichText::new(label).size(12.0).color(MUTED));
    ui.add(
        egui::Slider::new(value, range)
            .step_by(step)
            .suffix(format!(" {unit}")),
    )
    .changed()
}

fn metric(ui: &mut egui::Ui, label: &str, value: String, detail: &str, color: Color32) {
    ui.label(RichText::new(label).color(MUTED).size(10.0).strong());
    ui.label(RichText::new(value).size(26.0).color(color).strong());
    ui.label(RichText::new(detail).color(MUTED).size(10.0));
}

fn main() -> eframe::Result {
    eframe::run_native(
        "LIF",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1360.0, 820.0])
                .with_min_inner_size([1100.0, 720.0]),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(Lab::new(cc)))),
    )
}

fn load_chinese_font(ctx: &egui::Context) {
    let windows = std::env::var_os("WINDIR")
        .map(PathBuf::from)
        .map(|root| root.join("Fonts/msyh.ttc"));
    let candidates = [
        windows,
        Some(PathBuf::from(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        )),
        Some(PathBuf::from("/System/Library/Fonts/PingFang.ttc")),
    ];
    for path in candidates.into_iter().flatten() {
        if let Ok(bytes) = fs::read(path) {
            let mut fonts = egui::FontDefinitions::default();
            fonts
                .font_data
                .insert("chinese".into(), egui::FontData::from_owned(bytes).into());
            for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts
                    .families
                    .entry(family)
                    .or_default()
                    .insert(0, "chinese".into());
            }
            ctx.set_fonts(fonts);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lab() -> Lab {
        let parameters = Parameters::default();
        let result = Experiment::run(&parameters).unwrap();
        Lab {
            result_parameters: parameters.clone(),
            parameters,
            result,
            elapsed_ms: 0.0,
            view_ms: 200.0,
            status: "已就绪".into(),
        }
    }

    #[test]
    fn parameter_changes_recompute_actual_spikes() {
        let mut lab = lab();
        lab.parameters.current = 0.0;
        lab.recompute();
        assert_eq!(lab.result.count(), 0);
        lab.parameters.current = 2.0;
        lab.parameters.sinusoidal = true;
        lab.recompute();
        assert_eq!(lab.result.count(), 60);
    }

    #[test]
    fn renders_all_signals_at_supported_sizes() {
        for size in [[1100.0, 720.0], [1360.0, 820.0]] {
            let context = egui::Context::default();
            let mut lab = lab();
            let output = context.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(size[0], size[1]),
                    )),
                    ..Default::default()
                },
                |ctx| lab.render(ctx),
            );
            assert!(!output.shapes.is_empty());
            assert_eq!(lab.parameters, Parameters::default());
        }
    }

    #[test]
    fn duration_entry_preserves_valid_time_steps() {
        for duration in [250.0, 250.1] {
            let context = egui::Context::default();
            let mut lab = lab();
            lab.parameters.duration = duration;
            lab.recompute();
            for _ in 0..2 {
                let _ = context.run(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1360.0, 820.0),
                        )),
                        ..Default::default()
                    },
                    |ctx| lab.render(ctx),
                );
                assert_eq!(lab.parameters.duration, duration);
                assert_eq!(lab.result.spikes.len(), (duration / DT).round() as usize);
            }
        }
    }

    #[test]
    fn invalid_edit_preserves_consistent_result() {
        let mut lab = lab();
        let before = lab.parameters.clone();
        lab.parameters.duration = f64::NAN;
        lab.recompute();
        assert_eq!(lab.parameters, before);
        assert_eq!(lab.result.count(), 67);
    }

    #[test]
    fn exports_create_distinct_directories_and_complete_data() {
        let lab = lab();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test-export");
        let first = lab.export_to(&root).unwrap();
        let second = lab.export_to(&root).unwrap();
        assert_ne!(first, second);
        let csv = fs::read_to_string(first.join("trace.csv")).unwrap();
        assert_eq!(csv.lines().count(), 20_001);
        let metadata: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(first.join("parameters.json")).unwrap())
                .unwrap();
        assert_eq!(metadata["dt_ms"], 0.1);
        assert_eq!(metadata["parameters"]["current"], 2.0);
        assert!(second.join("trace.csv").exists());
    }
}
