import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, FileText, Brain, Check, Sparkles, Layers, Boxes, Database } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { AppShell } from "@/components/AppShell";
import { csrfToken } from "@/bootstrap";

// The heavy client-side extractors (PDF.js, Mammoth) are loaded globally by the
// mount shell's <script> tags — same as the legacy page — so we reference them
// off window here rather than bundling them.
declare global {
  interface Window {
    pdfjsLib: any;
    mammoth: any;
  }
}

export interface UploadData {
  error?: string;
  username: string;
  streak: number;
  xp: number;
}

const MAX_UPLOAD_SIZE = 4.5 * 1024 * 1024; // pptx server-side
const MAX_EXTRACT_SIZE = 50 * 1024 * 1024; // pdf/docx/txt client-side

const PIPELINE = [
  { label: "Uploading", icon: UploadCloud },
  { label: "Extracting Text", icon: FileText },
  { label: "Understanding Content", icon: Brain },
  { label: "Creating Chunks", icon: Layers },
  { label: "Generating Embeddings", icon: Boxes },
  { label: "Building Knowledge Base", icon: Database },
  { label: "Ready", icon: Check },
];

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
async function extractText(file: File): Promise<string | null> {
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "txt") return extractTxt(file);
  if (ext === "pdf") return extractPdf(file);
  if (ext === "docx") return extractDocx(file);
  return null;
}

// Keep a numeric input inside its allowed range. An empty or non-numeric field
// falls back to `fallback` so the hidden inputs always carry a valid value.
function clampInt(raw: string, min: number, max: number, fallback: number): number {
  const n = parseInt(raw, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

// Section-sampling truncation to stay under Vercel's payload cap while keeping
// document-wide coverage. Ported verbatim from the legacy upload.html.
function sampleText(text: string, maxChars = 150000): string {
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

export default function Upload({ data }: { data: UploadData }) {
  const [fileName, setFileName] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(data.error ?? "");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  // Quiz config lives in state, not in the DOM. The visible inputs unmount when
  // `busy` flips to show the pipeline, so the submitted values come from
  // always-mounted hidden inputs below — otherwise the fields would be missing
  // from the POST and the server would silently fall back to its defaults.
  const [numQuestions, setNumQuestions] = useState(10);
  const [difficulty, setDifficulty] = useState("medium");
  const [timerMinutes, setTimerMinutes] = useState(10);
  const fileRef = useRef<HTMLInputElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const extractedRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (window.pdfjsLib) {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/js/pdf.worker.min.js";
    }
  }, []);

  function pickFile(file: File): boolean {
    setError("");
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (ext === "pptx" && file.size > MAX_UPLOAD_SIZE) {
      setError("PowerPoint (.pptx) files are limited to 4.5 MB because they're processed on the server. Convert to PDF or compress it.");
      return false;
    }
    if (ext !== "pptx" && file.size > MAX_EXTRACT_SIZE) {
      setError("File is too large. Maximum supported size is 50 MB.");
      return false;
    }
    setFileName(file.name);
    return true;
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!file) {
      setError("Please select or drop a file first.");
      return;
    }
    const ext = file.name.split(".").pop()?.toLowerCase();
    setBusy(true);
    setStage(0);
    try {
      let text = "";
      if (["pdf", "docx", "txt"].includes(ext ?? "")) {
        setStage(1);
        text = (await extractText(file)) ?? "";
      }
      if (text && text.trim().length > 0) {
        // Advance the visual pipeline while the server generates.
        setStage(2);
        const sampled = sampleText(text, 150000);
        if (extractedRef.current) extractedRef.current.value = sampled;
        fileRef.current?.removeAttribute("required");
        if (fileRef.current) fileRef.current.value = "";
        setStage(3);
        setTimeout(() => setStage(4), 300);
        setTimeout(() => setStage(5), 600);
        setTimeout(() => formRef.current?.submit(), 200);
      } else {
        if (file.size > MAX_UPLOAD_SIZE) {
          setError("No readable text could be extracted, and the file exceeds 4.5 MB so it can't be uploaded directly. Try a text-based document.");
          setBusy(false);
          return;
        }
        setStage(2);
        setTimeout(() => formRef.current?.submit(), 200);
      }
    } catch (err: any) {
      if (file.size > MAX_UPLOAD_SIZE) {
        setError("Could not extract text: " + (err?.message ?? "") + ". File exceeds 4.5 MB, so it can't be uploaded directly. Try another document.");
        setBusy(false);
      } else {
        setStage(2);
        setTimeout(() => formRef.current?.submit(), 200);
      }
    }
  }

  return (
    <AppShell active="knowledge" username={data.username} streak={data.streak} xp={data.xp}>
      <div className="max-w-2xl mx-auto px-5 py-10">
        <div className="text-center mb-8">
          <p className="text-accent-2 text-sm font-medium mb-2 flex items-center justify-center gap-1.5">
            <Sparkles className="w-4 h-4" /> Ingest &amp; index
          </p>
          <h1 className="text-[clamp(1.8rem,3vw,2.4rem)] font-bold">Build Your Knowledge Base</h1>
          <p className="text-text-2 mt-2">Upload notes, books, presentations and study material. Your AI coach reads, understands, and turns it into a personalized learning journey.</p>
        </div>

        {error && (
          <div className="mb-4 px-4 py-3 rounded-md bg-danger/10 border border-danger/20 text-danger text-sm">{error}</div>
        )}

        <Card pad="lg">
          <form ref={formRef} action="/upload" method="post" encType="multipart/form-data" onSubmit={onSubmit}>
            <input type="hidden" name="csrf_token" value={csrfToken()} />
            <input ref={extractedRef} type="hidden" name="extracted_text" />
            {/* Always mounted so the config survives the switch to <Pipeline />. */}
            <input type="hidden" name="num_questions" value={numQuestions} />
            <input type="hidden" name="difficulty" value={difficulty} />
            <input type="hidden" name="timer" value={timerMinutes} />

            {!busy ? (
              <>
                <label
                  htmlFor="file-input"
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragOver(false);
                    const f = e.dataTransfer.files?.[0];
                    if (f && pickFile(f) && fileRef.current) fileRef.current.files = e.dataTransfer.files;
                  }}
                  className={`flex flex-col items-center justify-center text-center gap-3 p-10 rounded-lg border-2 border-dashed cursor-pointer transition-all ${
                    dragOver ? "border-accent bg-accent/5" : fileName ? "border-success/40 bg-success/[0.04]" : "border-white/[0.12] hover:border-white/25"
                  }`}
                >
                  <span className={`grid place-items-center w-14 h-14 rounded-full ${fileName ? "bg-success/10 text-success" : "bg-accent/10 text-accent"}`}>
                    {fileName ? <Check className="w-7 h-7" /> : <UploadCloud className="w-7 h-7" />}
                  </span>
                  <p className="text-text-2">
                    <span className="text-accent font-medium">Click to browse</span> or drag &amp; drop
                  </p>
                  <p className="text-text-3 text-sm">PDF, DOCX, PPTX or TXT · up to 50 MB</p>
                  {fileName && <p className="text-success text-sm font-medium mt-1">{fileName}</p>}
                  <input
                    ref={fileRef}
                    type="file"
                    name="file"
                    id="file-input"
                    accept=".pdf,.docx,.pptx,.txt"
                    required
                    className="hidden"
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) pickFile(f); }}
                  />
                </label>

                <div className="grid sm:grid-cols-3 gap-4 mt-5">
                  <div>
                    <label htmlFor="num-questions" className="text-text-2 text-sm block mb-1.5">Questions</label>
                    <input id="num-questions" type="number" min={1} max={30} value={numQuestions}
                      onChange={(e) => setNumQuestions(clampInt(e.target.value, 1, 30, 10))}
                      className="w-full h-11 px-3 rounded-md bg-inset border border-white/[0.08] text-text focus:outline-none focus:border-accent/50" />
                  </div>
                  <div>
                    <label htmlFor="difficulty" className="text-text-2 text-sm block mb-1.5">Difficulty</label>
                    <select id="difficulty" value={difficulty} onChange={(e) => setDifficulty(e.target.value)}
                      className="w-full h-11 px-3 rounded-md bg-inset border border-white/[0.08] text-text focus:outline-none focus:border-accent/50">
                      <option value="easy">Easy</option>
                      <option value="medium">Medium</option>
                      <option value="hard">Hard</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="timer" className="text-text-2 text-sm block mb-1.5">Time limit (minutes)</label>
                    <input id="timer" type="number" min={1} max={180} value={timerMinutes}
                      onChange={(e) => setTimerMinutes(clampInt(e.target.value, 1, 180, 10))}
                      className="w-full h-11 px-3 rounded-md bg-inset border border-white/[0.08] text-text focus:outline-none focus:border-accent/50" />
                  </div>
                </div>

                <Button type="submit" size="lg" className="w-full mt-6" leftIcon={<Sparkles className="w-4 h-4" />}>
                  Generate Practice
                </Button>
              </>
            ) : (
              <Pipeline stage={stage} />
            )}
          </form>
        </Card>
      </div>
    </AppShell>
  );
}

function Pipeline({ stage }: { stage: number }) {
  return (
    <div className="py-6">
      <div className="space-y-2">
        {PIPELINE.map((step, i) => {
          const Icon = step.icon;
          const done = i < stage;
          const active = i === stage;
          return (
            <div key={step.label} className="flex items-center gap-3">
              <span className={`grid place-items-center w-9 h-9 rounded-full transition-colors ${
                done ? "bg-success/15 text-success" : active ? "bg-accent/15 text-accent" : "bg-white/[0.04] text-text-3"
              }`}>
                {done ? <Check className="w-4 h-4" /> : <Icon className="w-4 h-4" />}
              </span>
              <span className={`text-sm ${active ? "text-text font-medium" : done ? "text-text-2" : "text-text-3"}`}>
                {step.label}
              </span>
              {active && (
                <motion.span
                  className="ml-auto w-4 h-4 border-2 border-accent/30 border-t-accent rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
                />
              )}
            </div>
          );
        })}
      </div>
      <AnimatePresence>
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center text-text-3 text-sm mt-6">
          Generating your practice set with AI…
        </motion.p>
      </AnimatePresence>
    </div>
  );
}
