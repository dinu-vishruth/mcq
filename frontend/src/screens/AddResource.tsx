import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, FileText, Brain, Check, Library, Layers, Boxes, Database, ArrowLeft } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { AppShell } from "@/components/AppShell";
import { csrfToken } from "@/bootstrap";
import { extractText, sampleText, validateFileSize, initPdfWorker, MAX_UPLOAD_SIZE } from "@/lib/extract";

export interface AddResourceData {
  error?: string;
  username: string;
  streak: number;
  xp: number;
}

// Store-only pipeline — no quiz-generation step, ends at "Saved".
const PIPELINE = [
  { label: "Uploading", icon: UploadCloud },
  { label: "Extracting Text", icon: FileText },
  { label: "Understanding Content", icon: Brain },
  { label: "Creating Chunks", icon: Layers },
  { label: "Generating Embeddings", icon: Boxes },
  { label: "Saving to Library", icon: Database },
  { label: "Saved", icon: Check },
];

export default function AddResource({ data }: { data: AddResourceData }) {
  const [fileName, setFileName] = useState("");
  const [title, setTitle] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(data.error ?? "");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const extractedRef = useRef<HTMLInputElement>(null);

  useEffect(() => { initPdfWorker(); }, []);

  function pickFile(file: File): boolean {
    setError("");
    const sizeErr = validateFileSize(file);
    if (sizeErr) { setError(sizeErr); return false; }
    setFileName(file.name);
    // Default the title to the filename (without extension) if not set yet.
    if (!title) setTitle(file.name.replace(/\.[^.]+$/, ""));
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
        // .pptx (or no client text) -> let the server extract it.
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
    <AppShell active="journey" username={data.username} streak={data.streak} xp={data.xp}>
      <div className="max-w-2xl mx-auto px-5 py-10">
        <a href="/journey" className="inline-flex items-center gap-1.5 text-text-3 hover:text-text text-sm mb-6">
          <ArrowLeft className="w-4 h-4" /> Back to Learning Journey
        </a>
        <div className="text-center mb-8">
          <p className="text-accent-2 text-sm font-medium mb-2 flex items-center justify-center gap-1.5">
            <Library className="w-4 h-4" /> Add to your library
          </p>
          <h1 className="text-[clamp(1.8rem,3vw,2.4rem)] font-bold">Save a Study Resource</h1>
          <p className="text-text-2 mt-2">
            Upload notes, books, or slides. We read, understand, and index them so you can generate quizzes from them anytime — this just saves it, no quiz yet.
          </p>
        </div>

        {error && (
          <div className="mb-4 px-4 py-3 rounded-md bg-danger/10 border border-danger/20 text-danger text-sm">{error}</div>
        )}

        <Card pad="lg">
          <form ref={formRef} action="/ingest_resource" method="post" encType="multipart/form-data" onSubmit={onSubmit}>
            <input type="hidden" name="csrf_token" value={csrfToken()} />
            <input ref={extractedRef} type="hidden" name="extracted_text" />
            <input type="hidden" name="title" value={title} />

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

                <div className="mt-5">
                  <label htmlFor="title" className="text-text-2 text-sm block mb-1.5">Resource name</label>
                  <input
                    id="title"
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="e.g. DBMS Unit 3 — Transactions"
                    className="w-full h-11 px-3 rounded-md bg-inset border border-white/[0.08] text-text placeholder:text-text-3 focus:outline-none focus:border-accent/50"
                  />
                </div>

                <Button type="submit" size="lg" className="w-full mt-6" leftIcon={<Library className="w-4 h-4" />}>
                  Save to Library
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
          Saving to your library…
        </motion.p>
      </AnimatePresence>
    </div>
  );
}
