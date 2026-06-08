import fs from "node:fs/promises";
import path from "node:path";

import {
  createSlideContext,
  ensureArtifactToolWorkspace,
  importArtifactTool,
  saveBlobToFile,
} from "/Users/song-yonghwi/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs";

const ROOT = "/Users/song-yonghwi/Documents/vehicle_trajectory_project";
const WORKSPACE =
  "/Users/song-yonghwi/Documents/vehicle_trajectory_project/outputs/019e832e-e60e-7980-a4ea-d88bb6991f7a/presentations/trajectory-extra-metrics";
const OUT_DIR = path.join(ROOT, "outputs/presentation");
const FINAL_PPTX = path.join(OUT_DIR, "vehicle_trajectory_forecasting_presentation_with_metrics.pptx");
const PREVIEW_DIR = path.join(WORKSPACE, "preview-final");
const LAYOUT_DIR = path.join(WORKSPACE, "layout-final");
const CONTACT_SHEET = path.join(PREVIEW_DIR, "contact-sheet.png");

const W = 1280;
const H = 720;

const colors = {
  bg: "#F5F8FC",
  ink: "#172033",
  muted: "#64748B",
  soft: "#E8EEF5",
  softer: "#FFFFFF",
  teal: "#149A8B",
  tealDark: "#0F766E",
  navy: "#1B365D",
  orange: "#F97316",
  red: "#DC2626",
  blue: "#2563EB",
  green: "#16A34A",
  purple: "#7C3AED",
  grayLine: "#D8E0EA",
};

const modelComparison = [
  { model: "Linear", ade: 1.5324, fde: 3.7973, minAde: 1.5324, minFde: 3.7973, miss: 0.5963, params: 0 },
  { model: "LSTM", ade: 0.7917, fde: 2.0570, minAde: 0.7917, minFde: 2.0570, miss: 0.3816, params: 401666 },
  { model: "Transformer", ade: 0.8515, fde: 2.1835, minAde: 0.8515, minFde: 2.1835, miss: 0.4001, params: 406332 },
  { model: "PCA Diffusion", ade: 1.4339, fde: 3.6640, minAde: 0.4175, minFde: 0.8583, miss: 0.6241, params: 222988 },
  { model: "Direct Diffusion", ade: 1.9282, fde: 4.9114, minAde: 0.5009, minFde: 0.9845, miss: 0.7267, params: 531132 },
];

const diffusionBeforeAfter = [
  { model: "PCA Diffusion", metric: "minADE", before: 6.0968, after: 0.4175, color: colors.teal },
  { model: "PCA Diffusion", metric: "minFDE", before: 11.6064, after: 0.8583, color: colors.tealDark },
  { model: "Direct Diffusion", metric: "minADE", before: 9.8817, after: 0.5009, color: colors.purple },
  { model: "Direct Diffusion", metric: "minFDE", before: 18.5069, after: 0.9845, color: "#5B21B6" },
].map((row) => ({
  ...row,
  improvement: ((row.before - row.after) / row.before) * 100,
}));

function line(fill = "#00000000", width = 0, style = "solid") {
  return { style, fill, width };
}

function addShape(ctx, slide, x, y, width, height, fill = colors.softer, stroke = colors.grayLine, strokeWidth = 1) {
  return ctx.addShape(slide, {
    x,
    y,
    width,
    height,
    fill,
    line: strokeWidth > 0 ? line(stroke, strokeWidth) : line("#00000000", 0),
  });
}

function addText(ctx, slide, text, x, y, width, height, opts = {}) {
  return ctx.addText(slide, {
    text,
    x,
    y,
    width,
    height,
    fontSize: opts.size ?? 24,
    color: opts.color ?? colors.ink,
    bold: opts.bold ?? false,
    align: opts.align ?? "left",
    valign: opts.valign ?? "top",
    typeface: opts.face ?? "Aptos",
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
    fill: opts.fill ?? "#00000000",
    line: opts.line ?? line("#00000000", 0),
  });
}

function addBase(ctx, presentation, number, section, title, subtitle = "") {
  const slide = presentation.slides.add();
  addShape(ctx, slide, 0, 0, W, H, colors.bg, colors.bg, 0);
  addShape(ctx, slide, 0, 0, W, 8, colors.teal, colors.teal, 0);
  addText(ctx, slide, `${String(number).padStart(2, "0")} / ${section}`, 58, 38, 420, 26, {
    size: 14,
    color: colors.teal,
    bold: true,
  });
  addText(ctx, slide, title, 58, 78, 900, 42, {
    size: 31,
    color: colors.ink,
    bold: true,
    face: "Aptos Display",
  });
  if (subtitle) {
    addText(ctx, slide, subtitle, 60, 128, 1080, 34, {
      size: 17,
      color: colors.muted,
    });
  }
  addText(ctx, slide, "Argoverse 2 Trajectory Forecasting Project", 982, 676, 240, 16, {
    size: 11,
    color: "#94A3B8",
    align: "right",
  });
  return slide;
}

function addKpi(ctx, slide, label, value, x, y, width, accent = colors.teal) {
  addShape(ctx, slide, x, y, width, 92, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, label, x + 18, y + 18, width - 36, 18, { size: 13, color: accent, bold: true });
  addText(ctx, slide, value, x + 18, y + 42, width - 36, 32, { size: 28, color: colors.ink, bold: true });
}

function addTable(ctx, slide, rows, columns, x, y, widths, rowHeight = 38, headerFill = colors.navy) {
  let cursorX = x;
  columns.forEach((col, index) => {
    addShape(ctx, slide, cursorX, y, widths[index], rowHeight, headerFill, colors.grayLine, 1);
    addText(ctx, slide, col, cursorX + 8, y + 10, widths[index] - 16, rowHeight - 16, {
      size: 12,
      color: "#FFFFFF",
      bold: true,
      align: index === 0 ? "left" : "right",
    });
    cursorX += widths[index];
  });

  rows.forEach((row, rowIndex) => {
    cursorX = x;
    const fill = rowIndex % 2 === 0 ? colors.softer : "#EEF3F8";
    columns.forEach((col, colIndex) => {
      addShape(ctx, slide, cursorX, y + rowHeight * (rowIndex + 1), widths[colIndex], rowHeight, fill, colors.grayLine, 1);
      addText(ctx, slide, String(row[col] ?? ""), cursorX + 8, y + rowHeight * (rowIndex + 1) + 10, widths[colIndex] - 16, rowHeight - 15, {
        size: 12,
        color: colIndex === 0 ? colors.ink : colors.muted,
        bold: row.highlight && colIndex === 0,
        align: colIndex === 0 ? "left" : "right",
      });
      cursorX += widths[colIndex];
    });
  });
}

function addBullets(ctx, slide, items, x, y, width, gap = 42) {
  items.forEach((item, index) => {
    const top = y + index * gap;
    addShape(ctx, slide, x, top + 5, 8, 8, item.color ?? colors.teal, item.color ?? colors.teal, 0);
    addText(ctx, slide, item.text, x + 22, top, width - 22, gap - 4, {
      size: item.size ?? 18,
      color: item.colorText ?? colors.ink,
      bold: item.bold ?? false,
    });
  });
}

function addHorizontalBar(ctx, slide, label, value, maxValue, x, y, width, fill, suffix = "", opts = {}) {
  const labelWidth = opts.labelWidth ?? 150;
  addText(ctx, slide, label, x, y + 2, labelWidth, 22, { size: 13, color: colors.ink, bold: opts.bold ?? false });
  addShape(ctx, slide, x + labelWidth, y, width, 18, "#E2E8F0", "#E2E8F0", 0);
  addShape(ctx, slide, x + labelWidth, y, Math.max(2, (value / maxValue) * width), 18, fill, fill, 0);
  addText(ctx, slide, `${value.toFixed(opts.decimals ?? 1)}${suffix}`, x + labelWidth + width + 12, y - 2, 90, 24, {
    size: 13,
    color: colors.muted,
  });
}

function addTwoValueBar(ctx, slide, row, x, y, maxValue) {
  const chartX = x + 160;
  addText(ctx, slide, row.model, x, y, 130, 20, { size: 13, color: colors.ink, bold: true });
  addText(ctx, slide, row.metric, x, y + 23, 130, 18, { size: 12, color: colors.muted });
  addShape(ctx, slide, chartX, y + 2, 370, 16, "#E2E8F0", "#E2E8F0", 0);
  addShape(ctx, slide, chartX, y + 2, (row.before / maxValue) * 370, 16, "#CBD5E1", "#CBD5E1", 0);
  addShape(ctx, slide, chartX, y + 27, 370, 16, "#E2E8F0", "#E2E8F0", 0);
  addShape(ctx, slide, chartX, y + 27, Math.max(2, (row.after / maxValue) * 370), 16, row.color, row.color, 0);
  addText(ctx, slide, `${row.before.toFixed(2)} → ${row.after.toFixed(2)}`, chartX + 388, y + 3, 120, 18, {
    size: 12,
    color: colors.muted,
  });
  addText(ctx, slide, `-${row.improvement.toFixed(1)}%`, chartX + 388, y + 27, 105, 20, {
    size: 14,
    color: row.color,
    bold: true,
  });
}

async function addImageCard(ctx, slide, imagePath, title, caption, x, y, width, height) {
  addShape(ctx, slide, x, y, width, height, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, title, x + 24, y + 18, width - 48, 24, {
    size: 17,
    bold: true,
    align: "center",
  });
  await ctx.addImage(slide, {
    path: imagePath,
    x: x + 34,
    y: y + 54,
    width: width - 68,
    height: height - 112,
    fit: "contain",
    alt: title,
  });
  addText(ctx, slide, caption, x + 24, y + height - 42, width - 48, 20, {
    size: 13,
    color: colors.muted,
    align: "center",
  });
}

async function makeSlide01(ctx, presentation) {
  const slide = addBase(ctx, presentation, 1, "Project Overview", "차량·보행자의 과거 5초 움직임으로 미래 3초 궤적을 예측한다.");
  addText(ctx, slide, "차량·보행자 미래 궤적 예측", 60, 198, 560, 58, {
    size: 40,
    color: colors.ink,
    bold: true,
    face: "Aptos Display",
  });
  addText(
    ctx,
    slide,
    "Linear, LSTM, Transformer, Direct Diffusion, PCA Diffusion을 같은 AV2 validation split에서 비교했다.",
    62,
    264,
    560,
    76,
    { size: 21, color: colors.muted },
  );
  addKpi(ctx, slide, "Input", "5s / 50 steps", 60, 388, 178);
  addKpi(ctx, slide, "Output", "3s / 30 steps", 258, 388, 178, colors.orange);
  addKpi(ctx, slide, "Val", "23,706 samples", 456, 388, 190, colors.blue);
  await ctx.addImage(slide, {
    path: path.join(ROOT, "outputs/presentation/figures/presentation_best_case_overlay.png"),
    x: 710,
    y: 188,
    width: 480,
    height: 330,
    fit: "contain",
    alt: "best trajectory prediction example",
  });
  addShape(ctx, slide, 688, 170, 524, 372, "#FFFFFF00", colors.grayLine, 1);
  addText(ctx, slide, "실제 도로 궤적 기반 예측 예시", 760, 548, 390, 22, {
    size: 14,
    color: colors.muted,
    align: "center",
  });
}

function makeSlide02(ctx, presentation) {
  const slide = addBase(ctx, presentation, 2, "Motivation", "자율주행에서 주변 객체의 미래 위치 예측은 안전한 경로 계획의 핵심이다.");
  const cards = [
    ["안전", "보행자·주변 차량의 감속, 회전, 차선 변경 가능성을 미리 예측해야 한다.", colors.teal],
    ["불확실성", "미래는 하나로 고정되지 않는다. 여러 가능한 경로를 함께 고려해야 한다.", colors.orange],
    ["모델 비교", "단순 baseline부터 sequence model, diffusion 후보 생성까지 같은 지표로 비교했다.", colors.blue],
  ];
  cards.forEach(([title, body, accent], index) => {
    const x = 72 + index * 388;
    addShape(ctx, slide, x, 220, 330, 230, colors.softer, colors.grayLine, 1);
    addShape(ctx, slide, x, 220, 330, 6, accent, accent, 0);
    addText(ctx, slide, title, x + 26, 252, 260, 34, { size: 25, color: colors.ink, bold: true });
    addText(ctx, slide, body, x + 26, 312, 270, 92, { size: 18, color: colors.muted });
  });
  addText(ctx, slide, "문제 정의: past trajectory 50 steps → future trajectory 30 steps", 154, 526, 972, 38, {
    size: 25,
    color: colors.ink,
    bold: true,
    align: "center",
  });
}

function makeSlide03(ctx, presentation) {
  const slide = addBase(ctx, presentation, 3, "Dataset & Preprocessing", "Argoverse 2 Motion Forecasting 데이터를 모델이 학습 가능한 시계열 텐서로 변환했다.");
  const steps = [
    ["Raw AV2", "parquet trajectory"],
    ["Normalize", "마지막 관측 시점 기준 상대좌표"],
    ["Feature", "x, y, velocity, heading"],
    ["Label", "future 30-step x, y"],
    ["Split", "train 189,541 / val 23,706"],
  ];
  steps.forEach(([title, body], index) => {
    const x = 70 + index * 230;
    addShape(ctx, slide, x, 245, 178, 126, colors.softer, colors.grayLine, 1);
    addText(ctx, slide, title, x + 14, 268, 150, 24, { size: 18, color: colors.ink, bold: true, align: "center" });
    addText(ctx, slide, body, x + 14, 306, 150, 42, { size: 13, color: colors.muted, align: "center" });
    if (index < steps.length - 1) {
      addText(ctx, slide, "→", x + 188, 286, 42, 42, { size: 28, color: colors.teal, bold: true, align: "center" });
    }
  });
  addShape(ctx, slide, 120, 472, 1040, 70, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "Leakage 방지", 148, 494, 140, 28, { size: 20, color: colors.teal, bold: true });
  addText(ctx, slide, "Scaler와 PCA는 train split에서만 fit하고 validation에는 transform만 적용했다.", 304, 496, 760, 28, {
    size: 19,
    color: colors.ink,
  });
}

function makeSlide04(ctx, presentation) {
  const slide = addBase(ctx, presentation, 4, "Algorithms", "단일 예측 모델과 후보 생성 모델을 함께 비교했다.");
  const rows = [
    ["Linear", "현재 속도 유지 가정", "가장 단순한 기준선"],
    ["LSTM", "순환 구조", "짧은 시계열 움직임 패턴 학습"],
    ["Transformer", "Attention", "과거 시점 간 관계 학습"],
    ["Direct Diffusion", "60D trajectory 직접 생성", "여러 미래 후보 생성"],
    ["PCA Diffusion", "PCA 기반 diffusion", "차원 축소로 안정적인 후보 생성"],
  ];
  rows.forEach(([model, method, note], index) => {
    const y = 190 + index * 78;
    addShape(ctx, slide, 94, y, 1090, 58, index % 2 === 0 ? colors.softer : "#EEF3F8", colors.grayLine, 1);
    addText(ctx, slide, model, 124, y + 15, 210, 24, { size: 18, bold: true, color: colors.ink });
    addText(ctx, slide, method, 370, y + 15, 300, 24, { size: 17, color: colors.teal, bold: true });
    addText(ctx, slide, note, 710, y + 15, 420, 24, { size: 16, color: colors.muted });
  });
}

function makeSlide05(ctx, presentation) {
  const slide = addBase(ctx, presentation, 5, "Implementation", "AV2 데이터부터 전처리, 학습, 평가, 시각화까지 하나의 파이프라인으로 구현했다.");
  const steps = [
    ["1", "Data Load", "AV2 데이터\n읽기", colors.teal],
    ["2", "Preprocess", "상대좌표·속도·\nheading feature 생성", colors.blue],
    ["3", "Dataset", "train/val .npz\n학습 포맷 저장", colors.orange],
    ["4", "Train", "Linear, LSTM,\nTransformer, Diffusion", colors.purple],
    ["5", "Evaluate", "ADE/FDE/Miss Rate\n및 시각화", colors.green],
  ];
  steps.forEach(([num, title, body, accent], index) => {
    const x = 62 + index * 236;
    addShape(ctx, slide, x, 222, 188, 180, colors.softer, colors.grayLine, 1);
    addShape(ctx, slide, x, 222, 188, 6, accent, accent, 0);
    addShape(ctx, slide, x + 18, 252, 34, 34, accent, accent, 0);
    addText(ctx, slide, num, x + 18, 258, 34, 22, {
      size: 17,
      color: "#FFFFFF",
      bold: true,
      align: "center",
    });
    addText(ctx, slide, title, x + 64, 254, 102, 26, {
      size: 18,
      color: colors.ink,
      bold: true,
    });
    addText(ctx, slide, body, x + 20, 316, 148, 52, {
      size: 14,
      color: colors.muted,
      align: "center",
    });
    if (index < steps.length - 1) {
      addText(ctx, slide, "→", x + 194, 290, 42, 36, {
        size: 26,
        color: colors.teal,
        bold: true,
        align: "center",
      });
    }
  });
  addShape(ctx, slide, 106, 466, 1068, 84, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "핵심 구현 기준", 138, 492, 150, 28, { size: 20, color: colors.teal, bold: true });
  addText(ctx, slide, "모든 모델이 같은 데이터 포맷과 같은 validation split을 사용하도록 맞추고, 같은 평가 지표로 비교했다.", 306, 494, 770, 26, {
    size: 18,
    color: colors.ink,
  });
  addText(ctx, slide, "전처리 → 모델별 학습 → 평가 표 생성 → 예측 궤적 그림 생성", 244, 592, 792, 34, {
    size: 24,
    color: colors.ink,
    bold: true,
    align: "center",
  });
}

function makeSlide06(ctx, presentation) {
  const slide = addBase(ctx, presentation, 6, "Experiment Settings", "모든 모델은 같은 val_full split에서 평가했고, GPU 학습 결과만 비교에 사용했다.");
  const rows = [
    { Model: "Linear", Setting: "현재 움직임 유지 가정", Epoch: "-", Batch: "-", Notes: "학습 없음" },
    { Model: "LSTM", Setting: "hidden 128 / 2 layers", Epoch: "30", Batch: "64", Notes: "Smooth L1" },
    { Model: "Transformer", Setting: "d_model 128 / 4 heads / 3 layers", Epoch: "30", Batch: "32", Notes: "Attention" },
    { Model: "PCA Diff.", Setting: "latent 12 / sampling 75 / K=16", Epoch: "30", Batch: "64", Notes: "selected pca_b" },
    { Model: "Direct Diff.", Setting: "trajectory 60D / sampling 75 / K=16", Epoch: "30", Batch: "16", Notes: "selected direct_f" },
  ];
  addTable(ctx, slide, rows, ["Model", "Setting", "Epoch", "Batch", "Notes"], 68, 202, [160, 420, 90, 90, 224], 42);
  addShape(ctx, slide, 122, 486, 1036, 72, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "공통 조건", 152, 510, 120, 28, { size: 20, bold: true, color: colors.teal });
  addText(ctx, slide, "입력 past 50 steps, 출력 future 30 steps, Miss Rate threshold 2.0m", 292, 511, 740, 28, {
    size: 19,
    color: colors.ink,
  });
}

function makeSlide07(ctx, presentation) {
  const slide = addBase(ctx, presentation, 7, "Experiment Results", "단일 예측은 LSTM, 후보 생성 best-of-K는 PCA Diffusion이 가장 낮은 오차를 보였다.");
  const rows = modelComparison.map((row) => ({
    Model: row.model,
    ADE: row.ade.toFixed(4),
    FDE: row.fde.toFixed(4),
    minADE: row.minAde.toFixed(4),
    minFDE: row.minFde.toFixed(4),
    highlight: row.model === "LSTM" || row.model === "PCA Diffusion",
  }));
  addTable(ctx, slide, rows, ["Model", "ADE", "FDE", "minADE", "minFDE"], 62, 214, [242, 122, 122, 134, 134], 38);
  addKpi(ctx, slide, "Best single prediction", "LSTM", 850, 214, 268);
  addText(ctx, slide, "ADE 0.7917 / FDE 2.0570", 868, 296, 224, 22, { size: 12, color: colors.muted });
  addKpi(ctx, slide, "Best best-of-K", "PCA Diffusion", 850, 352, 268, colors.orange);
  addText(ctx, slide, "minADE 0.4175 / minFDE 0.8583", 868, 434, 224, 22, { size: 12, color: colors.muted });
  addShape(ctx, slide, 62, 500, 1056, 78, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "해석", 92, 528, 70, 28, { size: 19, color: colors.teal, bold: true });
  addText(
    ctx,
    slide,
    "LSTM/Transformer는 하나의 궤적을 안정적으로 예측했고, Diffusion은 여러 가능한 미래 후보 중 실제와 가까운 후보를 생성했다.",
    176,
    522,
    880,
    36,
    { size: 18, color: colors.ink },
  );
  addText(ctx, slide, "주의: minADE/minFDE는 deterministic ADE/FDE와 직접 같은 의미로 비교하지 않음", 58, 676, 640, 16, {
    size: 11,
    color: "#94A3B8",
  });
}

function makeSlide08(ctx, presentation) {
  const slide = addBase(ctx, presentation, 8, "Diffusion Tuning", "5-epoch pilot에서 불안정했던 Diffusion은 설정 선택과 장기 학습 후 best-of-K 오차가 크게 낮아졌다.");
  const maxValue = 18.6;
  addText(ctx, slide, "Before: 5-epoch pilot, K=4", 226, 196, 260, 24, { size: 14, color: colors.muted, bold: true });
  addText(ctx, slide, "After: selected final, K=16", 226, 221, 260, 24, { size: 14, color: colors.teal, bold: true });
  diffusionBeforeAfter.forEach((row, index) => {
    addTwoValueBar(ctx, slide, row, 92, 270 + index * 78, maxValue);
  });
  addShape(ctx, slide, 812, 224, 330, 134, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "PCA Diffusion 선택값", 836, 248, 250, 24, { size: 17, color: colors.teal, bold: true });
  addText(ctx, slide, "latent 12 · sampling 75", 836, 290, 250, 22, { size: 15, color: colors.ink });
  addText(ctx, slide, "minADE 0.4175 / minFDE 0.8583", 836, 320, 250, 22, { size: 14, color: colors.muted });
  addShape(ctx, slide, 812, 386, 330, 134, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "Direct Diffusion 선택값", 836, 410, 250, 24, { size: 17, color: colors.purple, bold: true });
  addText(ctx, slide, "60D direct · sampling 75", 836, 452, 250, 22, { size: 15, color: colors.ink });
  addText(ctx, slide, "minADE 0.5009 / minFDE 0.9845", 836, 482, 250, 22, { size: 14, color: colors.muted });
  addShape(ctx, slide, 118, 594, 1036, 52, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "주의", 150, 611, 62, 22, { size: 16, color: colors.orange, bold: true });
  addText(ctx, slide, "동일 조건 ablation은 아니며, 튜닝·sampling 수·학습 길이가 함께 반영된 before/after 비교다.", 228, 611, 808, 22, {
    size: 16,
    color: colors.ink,
  });
}

function makeSlide09(ctx, presentation) {
  const slide = addBase(ctx, presentation, 9, "Miss Rate & Model Efficiency", "정확도만이 아니라 실패 비율과 파라미터 수를 함께 보면 모델 선택 기준이 더 분명해진다.");
  addText(ctx, slide, "Miss Rate (FDE > 2.0m, 낮을수록 좋음)", 78, 200, 430, 24, {
    size: 17,
    color: colors.ink,
    bold: true,
  });
  const maxMiss = 0.75;
  modelComparison.forEach((row, index) => {
    const fill = row.model === "LSTM" ? colors.teal : row.model.includes("Diffusion") ? colors.orange : colors.blue;
    addHorizontalBar(ctx, slide, row.model, row.miss * 100, maxMiss * 100, 78, 244 + index * 47, 310, fill, "%", {
      labelWidth: 145,
      decimals: 1,
      bold: row.model === "LSTM",
    });
  });

  addText(ctx, slide, "Parameter 수와 ADE", 710, 200, 360, 24, {
    size: 17,
    color: colors.ink,
    bold: true,
  });
  const rows = modelComparison.map((row) => ({
    Model: row.model,
    "Params": row.params === 0 ? "0" : `${Math.round(row.params / 1000)}k`,
    ADE: row.ade.toFixed(4),
    FDE: row.fde.toFixed(4),
    highlight: row.model === "LSTM",
  }));
  addTable(ctx, slide, rows, ["Model", "Params", "ADE", "FDE"], 676, 242, [190, 92, 96, 96], 39);
  addShape(ctx, slide, 90, 535, 492, 70, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "Best Miss Rate", 118, 555, 150, 20, { size: 14, color: colors.teal, bold: true });
  addText(ctx, slide, "LSTM 38.2%", 118, 579, 180, 24, { size: 23, color: colors.ink, bold: true });
  addText(ctx, slide, "Transformer는 40.0%로 근접", 310, 582, 220, 20, { size: 14, color: colors.muted });
  addShape(ctx, slide, 676, 535, 424, 70, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "효율 해석", 704, 555, 120, 20, { size: 14, color: colors.orange, bold: true });
  addText(ctx, slide, "PCA Diffusion은 파라미터가 적지만 single ADE는 LSTM보다 높고, Direct는 가장 무겁다.", 704, 580, 348, 20, {
    size: 13,
    color: colors.ink,
  });
}

async function makeSlide10(ctx, presentation) {
  const slide = addBase(ctx, presentation, 10, "Visual Evidence", "표만으로 보이지 않는 성공·실패·후보 생성 패턴을 실제 궤적 그림으로 확인했다.");
  await addImageCard(
    ctx,
    slide,
    path.join(ROOT, "outputs/presentation/figures/presentation_best_case_overlay.png"),
    "성공 사례",
    "여러 모델 예측이 실제 미래와 가까움",
    54,
    198,
    348,
    338,
  );
  await addImageCard(
    ctx,
    slide,
    path.join(ROOT, "outputs/presentation/figures/presentation_worst_case_overlay.png"),
    "어려운 사례",
    "일부 모델이 회전/방향 변화를 놓침",
    432,
    198,
    348,
    338,
  );
  await addImageCard(
    ctx,
    slide,
    path.join(ROOT, "outputs/presentation/figures/presentation_diffusion_candidates.png"),
    "Diffusion 후보",
    "여러 가능한 미래 중 좋은 후보를 생성",
    810,
    198,
    348,
    338,
  );
  addShape(ctx, slide, 250, 568, 740, 58, colors.softer, colors.grayLine, 1);
  const legend = [
    ["Past", colors.ink],
    ["Ground truth", "#F59E0B"],
    ["Linear", colors.blue],
    ["LSTM", colors.green],
    ["Transformer", colors.red],
    ["PCA Diff.", colors.orange],
    ["Direct Diff.", colors.purple],
  ];
  legend.forEach(([label, color], index) => {
    const x = 278 + (index % 4) * 142;
    const y = 586 + Math.floor(index / 4) * 24;
    addShape(ctx, slide, x, y + 6, 24, 4, color, color, 0);
    addText(ctx, slide, label, x + 34, y, 96, 18, { size: 12, color: colors.muted });
  });
}

function makeSlide11(ctx, presentation) {
  const slide = addBase(ctx, presentation, 11, "Discussion", "수업에서 배운 전처리, 학습, 평가, 비교 실험을 실제 자율주행 데이터에 end-to-end로 적용했다.");
  const left = [
    { text: "LSTM은 단일 궤적 예측에서 가장 안정적인 성능을 보였다.", color: colors.teal, bold: true },
    { text: "Transformer는 attention 구조를 사용했지만, map/interaction feature가 없어 장점이 제한됐다.", color: colors.blue },
    { text: "Diffusion은 단일 예측보다 여러 미래 후보 생성에서 의미가 있었다.", color: colors.orange },
  ];
  addShape(ctx, slide, 82, 204, 520, 260, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "핵심 결론", 112, 232, 180, 30, { size: 24, color: colors.ink, bold: true });
  addBullets(ctx, slide, left, 118, 294, 420, 56);
  addShape(ctx, slide, 688, 204, 500, 260, colors.softer, colors.grayLine, 1);
  addText(ctx, slide, "한계와 개선 방향", 718, 232, 260, 30, { size: 24, color: colors.ink, bold: true });
  addBullets(ctx, slide, [
    { text: "HD map, lane graph, traffic light 미사용", color: colors.red },
    { text: "주변 agent interaction 모델링 부족", color: colors.red },
    { text: "Diffusion 후보 선택을 위한 ranking/confidence 필요", color: colors.red },
  ], 724, 294, 390, 56);
  addText(ctx, slide, "최종 의의: 실제 AV2 데이터로 모델 구현·GPU 학습·정량 평가·시각 분석까지 수행", 126, 548, 1010, 34, {
    size: 24,
    color: colors.ink,
    bold: true,
    align: "center",
  });
}

async function makeContactSheet(previewPaths) {
  const { spawnSync } = await import("node:child_process");
  const result = spawnSync(
    "/Users/song-yonghwi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
    [
      "/Users/song-yonghwi/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/make_contact_sheet.py",
      "--output",
      CONTACT_SHEET,
      ...previewPaths,
    ],
    { encoding: "utf8" },
  );
  if (result.status !== 0) {
    throw new Error([result.stdout, result.stderr].filter(Boolean).join("\n"));
  }
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  await ensureArtifactToolWorkspace(WORKSPACE);
  const artifact = await importArtifactTool(WORKSPACE);
  const { Presentation, PresentationFile } = artifact;
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  const ctx = createSlideContext(artifact, {
    slideSize: { width: W, height: H },
    outputDir: OUT_DIR,
    assetDir: path.join(WORKSPACE, "assets"),
    workspaceDir: WORKSPACE,
  });

  await makeSlide01(ctx, presentation);
  makeSlide02(ctx, presentation);
  makeSlide03(ctx, presentation);
  makeSlide04(ctx, presentation);
  makeSlide05(ctx, presentation);
  makeSlide06(ctx, presentation);
  makeSlide07(ctx, presentation);
  makeSlide08(ctx, presentation);
  makeSlide09(ctx, presentation);
  await makeSlide10(ctx, presentation);
  makeSlide11(ctx, presentation);

  const previewPaths = [];
  for (let index = 0; index < presentation.slides.count; index += 1) {
    const slide = presentation.slides.getItem(index);
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    const previewPath = path.join(PREVIEW_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`);
    await saveBlobToFile(preview, previewPath);
    previewPaths.push(previewPath);
    try {
      const layout = await presentation.export({ slide, format: "layout" });
      await fs.writeFile(path.join(LAYOUT_DIR, `slide-${String(index + 1).padStart(2, "0")}.layout.json`), await layout.text(), "utf8");
    } catch {
      // Layout export is useful for QA but not required for final PPTX generation.
    }
  }

  await makeContactSheet(previewPaths);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(JSON.stringify({ finalPptx: FINAL_PPTX, contactSheet: CONTACT_SHEET, slideCount: presentation.slides.count }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
