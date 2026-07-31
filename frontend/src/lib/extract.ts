// Client-side document text extraction, shared by the quiz-upload flow and the
// store-only "add resource" flow. Extracting in the browser avoids Vercel's
// 4.5 MB server upload cap for PDF/DOCX/TXT (only .pptx must go to the server).
//
// The heavy extractors (PDF.js, Mammoth) are loaded globally by the mount
// shell's <script> tags, so we reference them off window rather than bundling.

declare global {
  interface Window {
    pdfjsLib: any;
    mammoth: any;
  }
}

export const MAX_UPLOAD_SIZE = 4.5 * 1024 * 1024; // .pptx processed server-side
export const MAX_EXTRACT_SIZE = 50 * 1024 * 1024; // pdf/docx/txt client-side

export function initPdfWorker() {
  if (typeof window !== "undefined" && window.pdfjsLib) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/js/pdf.worker.min.js";
  }
}

async function extractTxt(file: File): Promise<string> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = (e) => res(e.target?.result as string);
    r.onerror = rej;
    r.readAsText(file);
  });
}

async function extractPdf(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const pdf = await window.pdfjsLib.getDocument({ data: buf }).promise;
  let text = "";
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const tc = await page.getTextContent();
    text += tc.items.map((it: any) => it.str).join(" ") + "\n";
  }
  return text;
}

async function extractDocx(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const r = await window.mammoth.extractRawText({ arrayBuffer: buf });
  return r.value;
}

export async function extractText(file: File): Promise<string | null> {
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "txt") return extractTxt(file);
  if (ext === "pdf") return extractPdf(file);
  if (ext === "docx") return extractDocx(file);
  return null; // .pptx (or unknown) — handled server-side
}

// Section-sampling truncation to stay under the payload cap while keeping
// document-wide coverage. Ported verbatim from the legacy upload.html.
export function sampleText(text: string, maxChars = 150000): string {
  if (text.length <= maxChars) return text;
  const numSections = 30;
  const charsPerSection = Math.floor(maxChars / numSections);
  const sectionSize = text.length / numSections;
  const chunks: string[] = [];
  for (let i = 0; i < numSections; i++) {
    const startIdx = Math.floor(i * sectionSize);
    let s = text.indexOf("\n", startIdx);
    if (s === -1 || s > startIdx + 1000) s = text.indexOf(" ", startIdx);
    if (s === -1 || s >= text.length) s = startIdx;
    else s += 1;
    let endIdx = s + charsPerSection;
    if (endIdx > text.length) endIdx = text.length;
    let e = text.indexOf("\n", endIdx);
    if (e === -1 || e > endIdx + 1000) e = text.indexOf(" ", endIdx);
    if (e === -1 || e > text.length) e = endIdx;
    const chunk = text.substring(s, e).trim();
    if (chunk.length) chunks.push(chunk);
  }
  return chunks.join("\n\n... [section transition] ...\n\n");
}

// Validate a picked file's size against the per-type limits. Returns an error
// string, or "" when the file is acceptable.
export function validateFileSize(file: File): string {
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "pptx" && file.size > MAX_UPLOAD_SIZE) {
    return "PowerPoint (.pptx) files are limited to 4.5 MB because they're processed on the server. Convert to PDF or compress it.";
  }
  if (ext !== "pptx" && file.size > MAX_EXTRACT_SIZE) {
    return "File is too large. Maximum supported size is 50 MB.";
  }
  return "";
}
