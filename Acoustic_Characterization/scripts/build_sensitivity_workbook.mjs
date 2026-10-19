import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const localConfigPath = process.env.CDSD_CONFIG_FILE
  ? path.resolve(process.env.CDSD_CONFIG_FILE)
  : path.join(root, "config", "paths.local.json");
let localConfig = {};
try {
  localConfig = JSON.parse(await fs.readFile(localConfigPath, "utf8"));
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
const configuredOutputRoot = process.env.CDSD_OUTPUT_ROOT || localConfig.CDSD_OUTPUT_ROOT;
const outputRoot = configuredOutputRoot ? path.resolve(configuredOutputRoot) : root;
const inputPath = path.join(outputRoot, "results", "child_adult_sensitivity_data.json");
const outputPath = path.join(outputRoot, "results", "child_adult_sensitivity.xlsx");
const previewDir = path.join(outputRoot, "results", "workbook_preview");
const data = JSON.parse(await fs.readFile(inputPath, "utf8"));

const workbook = Workbook.create();
const sheets = [
  ["correlations_all_adult", data.correlations],
  ["child_adult_distributions", data.distributions],
  ["segment_success_rates", data.successRates],
  ["absolute_f0_qc", data.absoluteF0Qc],
  ["child_speakers", data.childSpeakers],
  ["feature_provenance", data.featureProvenance],
];

const headerFill = "#1F4E78";
const headerFont = "#FFFFFF";
const stripeFill = "#EAF2F8";
for (const [name, rows] of sheets) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  if (!rows.length) continue;
  const headers = Object.keys(rows[0]);
  const values = [headers, ...rows.map((row) => headers.map((key) => row[key] ?? null))];
  const endCol = columnName(headers.length);
  const endRow = values.length;
  sheet.getRange(`A1:${endCol}${endRow}`).values = values;
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: headerFill,
    font: { bold: true, color: headerFont },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "medium", color: "#163A5C" },
  };
  sheet.getRange(`A1:${endCol}1`).format.rowHeight = 32;
  sheet.getRange(`A2:${endCol}${endRow}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
  };
  for (let row = 2; row <= endRow; row += 2) {
    sheet.getRange(`A${row}:${endCol}${row}`).format.fill = stripeFill;
  }
  sheet.getRange(`A1:${endCol}${endRow}`).format.autofitColumns();
  sheet.getRange(`A1:${endCol}${endRow}`).format.autofitRows();
  sheet.getRange(`A1:${endCol}${endRow}`).format.wrapText = false;
  sheet.freezePanes.freezeRows(1);
  const table = sheet.tables.add(`A1:${endCol}${endRow}`, true, `${name.replace(/[^A-Za-z0-9]/g, "")}Table`);
  table.style = "TableStyleMedium2";
  applyFormats(sheet, headers, endRow);
}

await fs.mkdir(previewDir, { recursive: true });
for (const [name] of sheets) {
  const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale: 1.5, format: "png" });
  await fs.writeFile(path.join(previewDir, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const inspected = await workbook.inspect({ kind: "sheet,table", maxChars: 4000, tableMaxRows: 5, tableMaxCols: 8 });
console.log(inspected.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" });
console.log(errors.ndjson);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

function columnName(count) {
  let n = count;
  let out = "";
  while (n > 0) {
    n--;
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26);
  }
  return out;
}

function applyFormats(sheet, headers, endRow) {
  headers.forEach((header, index) => {
    const col = columnName(index + 1);
    const range = sheet.getRange(`${col}2:${col}${endRow}`);
    if (/^n_|^n$|count|segments/i.test(header)) range.format.numberFormat = "#,##0";
    else if (/rate/i.test(header)) range.format.numberFormat = "0.00%";
    else if (/rho|^p_|^q_|mean|median|^sd$|^min$|^max$/i.test(header)) range.format.numberFormat = "0.000";
    if (/speaker_id|feature|group|direction|status/i.test(header)) range.format.horizontalAlignment = "left";
    else range.format.horizontalAlignment = "right";
    const maxWidth = /conclusion_changed/i.test(header) ? 42
      : /analysis_role/i.test(header) ? 42
      : /source/i.test(header) ? 34
      : /unit_or_scale/i.test(header) ? 24
      : /n_segments_with_upper_bound_hit/i.test(header) ? 32
      : /mean_upper_bound_hit_rate/i.test(header) ? 28
      : /feature|status|direction|changed/i.test(header) ? 24 : 18;
    range.format.columnWidth = maxWidth;
  });
}
