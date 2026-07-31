import { useEffect, useState } from "react";
import {
  FileQuestion, Loader2, Sparkles, Library, ChevronRight, UploadCloud,
  Clock, Hash, Gauge, Plus,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { csrfToken, apiGet, apiSend } from "@/bootstrap";
import type { KnowledgeItem } from "@/types";

interface MakeQuizData { username: string; document_id: number | null }

const DIFFS = ["easy", "medium", "hard"] as const;
const COUNTS = [5, 10, 15, 20, 25, 30];
const TIMERS = [
  { label: "5 min", value: 5 },
  { label: "10 min", value: 10 },
  { label: "20 min", value: 20 },
  { label: "30 min", value: 30 },
  { label: "60 min", value: 60 },
];

function qs(name: string): string | null {
  return new URLSearchParams(window.location.search).get(name);
}

// Make Quiz & Test: the single place to generate a quiz/test. Two paths —
// (1) upload fresh material (hands off to /upload, which generates immediately),
// (2) generate from a resource already saved in the Learning Journey library.
// Absorbs the old standalone Practice page.
export default function MakeQuiz({ data }: { data: MakeQuizData }) {
  const [items, setItems] = useState<KnowledgeItem[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [docId, setDocId] = useState<number | null>(
    data.document_id ?? (qs("doc") ? Number(qs("doc")) : null),
  );

  const [difficulty, setDifficulty] = useState<(typeof DIFFS)[number]>("medium");
  const [count, setCount] = useState<number>(Number(qs("count")) || 10);
  const [topic, setTopic] = useState(qs("topic") || "");
  const [timerMinutes, setTimerMinutes] = useState<number>(10);

  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    apiGet<{ items: KnowledgeItem[] }>("/api/knowledge")
      .then((d) => {
        const ready = d.items.filter((k) => k.indexed);
        setItems(ready);
        if (!docId && ready.length === 1) setDocId(ready[0].id);
      })
      .catch(() => setItems([]))
      .finally(() => setLoadingList(false));
  }, []); // eslint-disable-line

  async function generate() {
    if (!docId) { setError("Choose a saved resource first, or upload new material above."); return; }
    setGenerating(true);
    setError("");
    try {
      const res = await apiSend<{ session_key: string }>("/api/practice/generate", "POST", {
        document_id: docId,
        num_questions: count,
        difficulty,
        topic: topic.trim() || undefined,
        timer: timerMinutes * 60,
      });
      const form = document.createElement("form");
      form.method = "POST";
      form.action = "/student_login";
      form.innerHTML =
        `<input name="csrf_token" value="${csrfToken()}">` +
        `<input name="session_key" value="${res.session_key}">`;
      document.body.appendChild(form);
      form.submit();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed. Try again.");
      setGenerating(false);
    }
  }

  return (
    <AppShell active="knowledge" username={data.username}>
      <div className="max-w-3xl mx-auto px-5 lg:px-8 py-8">
        <h1 className="font-display text-2xl font-semibold flex items-center gap-2.5">
          <FileQuestion className="w-6 h-6 text-accent" /> Make Quiz &amp; Test
        </h1>
        <p className="text-text-2 mt-1.5 mb-8">Turn your material into a quiz — upload something new, or use a resource you've already saved.</p>

        {/* Path 1: upload fresh material -> generates immediately via /upload */}
        <a href="/upload" className="block mb-6">
          <Card hover pad="md" className="flex items-center gap-4">
            <span className="grid place-items-center w-11 h-11 rounded-md bg-accent/10 text-accent shrink-0">
              <UploadCloud className="w-5 h-5" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="font-display font-semibold flex items-center gap-1.5">
                Upload new material <ChevronRight className="w-4 h-4 text-text-3" />
              </div>
              <p className="text-text-2 text-sm mt-0.5">Upload a file and generate a quiz from it right away.</p>
            </div>
          </Card>
        </a>

        <div className="flex items-center gap-3 text-text-3 text-xs uppercase tracking-wider mb-6">
          <span className="h-px flex-1 bg-white/[0.08]" /> or use a saved resource <span className="h-px flex-1 bg-white/[0.08]" />
        </div>

        {/* Path 2: generate from a saved Learning Journey resource */}
        <Card pad="md" className="mb-4">
          <label className="flex items-center gap-2 text-sm font-medium mb-3"><Library className="w-4 h-4 text-accent" /> Saved resource</label>
          {loadingList ? (
            <div className="text-text-3 flex items-center gap-2 text-sm"><Loader2 className="w-4 h-4 animate-spin" /> Loading…</div>
          ) : items.length === 0 ? (
            <div className="text-text-2 text-sm flex items-center gap-2">
              No saved resources yet.
              <a href="/add_resource" className="text-accent hover:underline inline-flex items-center gap-1">
                <Plus className="w-3.5 h-3.5" /> Add one to your library
              </a>
            </div>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2">
              {items.map((k) => (
                <button key={k.id} onClick={() => setDocId(k.id)} className="text-left">
                  <div className={`px-3.5 py-3 rounded-md border text-sm transition-colors ${
                    docId === k.id ? "border-accent bg-accent/8 text-text" : "border-white/10 text-text-2 hover:bg-white/[0.03]"}`}>
                    <div className="font-medium truncate">{k.title}</div>
                    <div className="text-text-3 text-xs mt-0.5">{k.topic_count} topics</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </Card>

        <div className="grid gap-4 sm:grid-cols-3">
          <Setting icon={<Gauge className="w-4 h-4 text-accent" />} label="Difficulty">
            <Segment options={DIFFS.map((d) => ({ label: d[0].toUpperCase() + d.slice(1), value: d }))}
              value={difficulty} onChange={(v) => setDifficulty(v as any)} />
          </Setting>
          <Setting icon={<Hash className="w-4 h-4 text-accent" />} label="Questions">
            <Segment options={COUNTS.map((c) => ({ label: String(c), value: c }))}
              value={count} onChange={(v) => setCount(v as number)} />
          </Setting>
          <Setting icon={<Clock className="w-4 h-4 text-accent" />} label="Time limit">
            <Segment options={TIMERS.map((t) => ({ label: t.label, value: t.value }))}
              value={timerMinutes} onChange={(v) => setTimerMinutes(v as number)} />
          </Setting>
        </div>

        <Setting icon={<Sparkles className="w-4 h-4 text-accent" />} label="Topic focus (optional)" className="mt-4">
          <input
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g. Deadlocks, Normalization…"
            className="w-full h-11 px-3.5 rounded-md bg-inset border border-white/10 text-sm placeholder:text-text-3 focus:outline-none focus:border-accent/60"
          />
        </Setting>

        {error && <div className="mt-4 text-danger text-sm bg-danger/8 border border-danger/20 rounded-md px-4 py-3">{error}</div>}

        <div className="mt-6">
          <Button size="lg" className="w-full" onClick={generate} disabled={generating || !docId}
            leftIcon={generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            rightIcon={!generating ? <ChevronRight className="w-4 h-4" /> : undefined}>
            {generating ? "Generating your quiz…" : "Generate & Start"}
          </Button>
        </div>
      </div>
    </AppShell>
  );
}

function Setting({ icon, label, children, className = "" }: {
  icon: React.ReactNode; label: string; children: React.ReactNode; className?: string;
}) {
  return (
    <div className={className}>
      <label className="flex items-center gap-2 text-sm font-medium mb-2">{icon} {label}</label>
      {children}
    </div>
  );
}

function Segment<T extends string | number>({ options, value, onChange }: {
  options: { label: string; value: T }[]; value: T; onChange: (v: T) => void;
}) {
  return (
    <div className="flex gap-1.5 flex-wrap">
      {options.map((o) => (
        <button
          key={String(o.value)}
          onClick={() => onChange(o.value)}
          className={`px-3.5 h-10 rounded-md text-sm border transition-colors ${
            value === o.value ? "border-accent bg-accent/10 text-text font-medium" : "border-white/10 text-text-2 hover:bg-white/[0.03]"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
