import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Compass, BookOpen, Dumbbell, RefreshCw, MessageSquare, ClipboardCheck,
  Target, ChevronRight, Loader2, Library,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/Card";
import { apiGet } from "@/bootstrap";
import type { KnowledgeItem } from "@/types";

interface JourneyData { username: string; document_id: number | null; title: string }

interface Workflow {
  key: string;
  label: string;
  desc: string;
  icon: typeof BookOpen;
  tone: string;
  href: (docId: number | null) => string;
}

const WORKFLOWS: Workflow[] = [
  { key: "learn", label: "Learn Concepts", desc: "Build understanding with guided explanations from your source.",
    icon: BookOpen, tone: "text-accent bg-accent/10",
    href: (d) => (d ? `/practice?doc=${d}&mode=learn` : "/practice") },
  { key: "practice", label: "Practice MCQs", desc: "Configure and generate a practice set to test recall.",
    icon: Dumbbell, tone: "text-success bg-success/10",
    href: (d) => (d ? `/practice?doc=${d}` : "/practice") },
  { key: "revision", label: "Quick Revision", desc: "A fast, focused pass over the key points.",
    icon: RefreshCw, tone: "text-violet bg-violet/10",
    href: (d) => (d ? `/practice?doc=${d}&mode=revision&count=5` : "/practice") },
  { key: "interview", label: "Interview Preparation", desc: "Rapid-fire questions to sharpen on-the-spot recall.",
    icon: MessageSquare, tone: "text-warning bg-warning/10",
    href: (d) => (d ? `/practice?doc=${d}&mode=interview` : "/practice") },
  { key: "mock", label: "Mock Test", desc: "A timed, full-length set that mirrors the real thing.",
    icon: ClipboardCheck, tone: "text-accent bg-accent/10",
    href: (d) => (d ? `/practice?doc=${d}&mode=mock&count=20` : "/practice") },
  { key: "weak", label: "Weak Topic Practice", desc: "Target the concepts you miss most.",
    icon: Target, tone: "text-danger bg-danger/10",
    href: () => "/weak-topics" },
];

export default function Journey({ data }: { data: JourneyData }) {
  const [docId, setDocId] = useState<number | null>(data.document_id);
  const [docTitle, setDocTitle] = useState(data.title);
  const [items, setItems] = useState<KnowledgeItem[]>([]);
  const [loading, setLoading] = useState(!data.document_id);

  useEffect(() => {
    if (data.document_id) return; // already have a source
    apiGet<{ items: KnowledgeItem[] }>("/api/knowledge")
      .then((d) => setItems(d.items))
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, [data.document_id]);

  // No source chosen yet: let the user pick one first.
  if (!docId) {
    return (
      <AppShell active="journey" username={data.username}>
        <div className="max-w-4xl mx-auto px-5 lg:px-8 py-8">
          <h1 className="font-display text-2xl font-semibold flex items-center gap-2.5">
            <Compass className="w-6 h-6 text-accent" /> Learning Journey
          </h1>
          <p className="text-text-2 mt-1.5 mb-8">Pick a knowledge source to begin.</p>

          {loading ? (
            <div className="grid place-items-center py-20 text-text-3"><Loader2 className="w-6 h-6 animate-spin" /></div>
          ) : items.length === 0 ? (
            <Card pad="lg" className="text-center py-14">
              <div className="grid place-items-center w-12 h-12 rounded-lg bg-accent/10 text-accent mx-auto mb-3"><Library className="w-6 h-6" /></div>
              <h3 className="font-display font-semibold">No sources yet</h3>
              <p className="text-text-2 mt-1">Add a study resource to start a journey.</p>
              <a href="/upload" className="text-accent hover:underline mt-3 inline-block">Go to Knowledge →</a>
            </Card>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2">
              {items.map((k) => (
                <button
                  key={k.id}
                  onClick={() => { setDocId(k.id); setDocTitle(k.title); }}
                  className="text-left"
                >
                  <Card hover pad="md" className="flex items-center justify-between">
                    <div className="min-w-0">
                      <div className="font-medium truncate">{k.title}</div>
                      <div className="text-text-3 text-xs mt-0.5">{k.topic_count} topics · ~{k.est_minutes} min</div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-text-3 shrink-0" />
                  </Card>
                </button>
              ))}
            </div>
          )}
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell active="journey" username={data.username}>
      <div className="max-w-4xl mx-auto px-5 lg:px-8 py-8">
        {docTitle && (
          <div className="mb-2 text-text-3 text-sm flex items-center gap-1.5">
            <Library className="w-4 h-4" /> {docTitle}
          </div>
        )}
        <h1 className="font-display text-2xl font-semibold">What would you like to achieve today?</h1>
        <p className="text-text-2 mt-1.5 mb-8">Choose a workflow. Your coach will guide the session.</p>

        <div className="grid gap-4 sm:grid-cols-2">
          {WORKFLOWS.map((w, i) => {
            const Icon = w.icon;
            return (
              <motion.a
                key={w.key}
                href={w.href(docId)}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <Card hover pad="md" className="h-full flex items-start gap-4">
                  <span className={`grid place-items-center w-11 h-11 rounded-md shrink-0 ${w.tone}`}>
                    <Icon className="w-5 h-5" />
                  </span>
                  <div className="min-w-0">
                    <div className="font-display font-semibold flex items-center gap-1.5">
                      {w.label} <ChevronRight className="w-4 h-4 text-text-3" />
                    </div>
                    <p className="text-text-2 text-sm mt-1 leading-relaxed">{w.desc}</p>
                  </div>
                </Card>
              </motion.a>
            );
          })}
        </div>
      </div>
    </AppShell>
  );
}
