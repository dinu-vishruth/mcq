import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Dumbbell, Loader2, Sparkles, Library, ChevronRight, Zap, Clock, Hash, Gauge, Type,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { csrfToken, apiGet, apiSend } from "@/bootstrap";
import type { KnowledgeItem } from "@/types";

interface PracticeData { username: string; document_id: number | null }

const DIFFS = ["easy", "medium", "hard"] as const;
const COUNTS = [5, 10, 15, 20];
const TIMERS = [
  { label: "No timer", value: 3600 },
  { label: "1 min/q", value: 60 },
  { label: "1.5 min/q", value: 90 },
  { label: "2 min/q", value: 120 },
];
const TYPES = ["Multiple choice"];

function qs(name: string): string | null {
  return new URLSearchParams(window.location.search).get(name);
}

export default function Practice({ data }: { data: PracticeData }) {
  const [items, setItems] = useState<KnowledgeItem[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [docId, setDocId] = useState<number | null>(data.document_id);

  const [difficulty, setDifficulty] = useState<(typeof DIFFS)[number]>("medium");
  const [count, setCount] = useState<number>(Number(qs("count")) || 10);
  const [topic, setTopic] = useState("");
  const [adaptive, setAdaptive] = useState(true);
  const [timer, setTimer] = useState<number>(90);
  const [qtype] = useState(TYPES[0]);

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
    if (!docId) { setError("Choose a knowledge source first."); return; }
    setGenerating(true);
    setError("");
    try {
      const res = await apiSend<{ session_key: string }>("/api/practice/generate", "POST", {
        document_id: docId,
        num_questions: count,
        difficulty,
        topic: topic.trim() || undefined,
        timer,
      });
      // Hand off to the existing quiz flow via the classic form POST it expects.
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
    <AppShell active="practice" username={data.username}>
      <div className="max-w-3xl mx-auto px-5 lg:px-8 py-8">
        <h1 className="font-display text-2xl font-semibold flex items-center gap-2.5">
          <Dumbbell className="w-6 h-6 text-accent" /> Practice
        </h1>
        <p className="text-text-2 mt-1.5 mb-8">Tune the session, then generate questions from your knowledge.</p>

        {/* Source */}
        <Card pad="md" className="mb-4">
          <label className="flex items-center gap-2 text-sm font-medium mb-3"><Library className="w-4 h-4 text-accent" /> Knowledge source</label>
          {loadingList ? (
            <div className="text-text-3 flex items-center gap-2 text-sm"><Loader2 className="w-4 h-4 animate-spin" /> Loading…</div>
          ) : items.length === 0 ? (
            <div className="text-text-2 text-sm">
              No indexed sources. <a href="/upload" className="text-accent hover:underline">Add one →</a>
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

        <div className="grid gap-4 sm:grid-cols-2">
          <Setting icon={<Gauge className="w-4 h-4 text-accent" />} label="Difficulty">
            <Segment options={DIFFS.map((d) => ({ label: d[0].toUpperCase() + d.slice(1), value: d }))}
              value={difficulty} onChange={(v) => setDifficulty(v as any)} />
          </Setting>

          <Setting icon={<Hash className="w-4 h-4 text-accent" />} label="Questions">
            <Segment options={COUNTS.map((c) => ({ label: String(c), value: c }))}
              value={count} onChange={(v) => setCount(v as number)} />
          </Setting>

          <Setting icon={<Clock className="w-4 h-4 text-accent" />} label="Timer">
            <Segment options={TIMERS.map((t) => ({ label: t.label, value: t.value }))}
              value={timer} onChange={(v) => setTimer(v as number)} />
          </Setting>

          <Setting icon={<Type className="w-4 h-4 text-accent" />} label="Question type">
            <Segment options={TYPES.map((t) => ({ label: t, value: t }))} value={qtype} onChange={() => {}} />
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

        <button
          onClick={() => setAdaptive((a) => !a)}
          className="mt-4 w-full flex items-center justify-between px-4 py-3.5 rounded-md bg-card border border-white/[0.07] text-left"
        >
          <span className="flex items-center gap-2.5">
            <Zap className={`w-4 h-4 ${adaptive ? "text-violet" : "text-text-3"}`} />
            <span>
              <span className="text-sm font-medium">Adaptive mode</span>
              <span className="block text-text-3 text-xs">Weights questions toward your weaker topics</span>
            </span>
          </span>
          <span className={`w-10 h-6 rounded-full p-0.5 transition-colors ${adaptive ? "bg-violet" : "bg-white/10"}`}>
            <motion.span layout className="block w-5 h-5 rounded-full bg-white" style={{ marginLeft: adaptive ? "1rem" : 0 }} />
          </span>
        </button>

        {error && <div className="mt-4 text-danger text-sm bg-danger/8 border border-danger/20 rounded-md px-4 py-3">{error}</div>}

        <div className="mt-6">
          <Button size="lg" className="w-full" onClick={generate} disabled={generating || !docId}
            leftIcon={generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            rightIcon={!generating ? <ChevronRight className="w-4 h-4" /> : undefined}>
            {generating ? "Generating your set…" : "Generate & Start"}
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
