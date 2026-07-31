import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Compass, Plus, Dumbbell, Trash2, Clock, Layers, CheckCircle2, Loader2, Library,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { apiGet, apiSend } from "@/bootstrap";
import type { KnowledgeItem } from "@/types";

// Learning Journey is the user's saved-resource library. It stores what they
// upload (via /add_resource) and lets them turn any saved resource into a quiz.
// It does NOT generate anything itself and never redirects to a quiz on its own.
export default function Journey({ data }: { data: { username: string } }) {
  const [items, setItems] = useState<KnowledgeItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<number | null>(null);

  useEffect(() => {
    apiGet<{ items: KnowledgeItem[] }>("/api/knowledge")
      .then((d) => setItems(d.items))
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, []);

  async function remove(id: number) {
    if (!confirm("Remove this resource from your library? Your quiz results are kept.")) return;
    setBusy(id);
    try {
      await apiSend(`/api/knowledge/${id}`, "DELETE");
      setItems((prev) => prev.filter((k) => k.id !== id));
    } catch {
      alert("Could not remove. Please try again.");
    } finally {
      setBusy(null);
    }
  }

  return (
    <AppShell active="journey" username={data.username}>
      <div className="max-w-6xl mx-auto px-5 lg:px-8 py-8">
        <div className="flex items-start justify-between gap-4 mb-8">
          <div>
            <h1 className="font-display text-2xl font-semibold flex items-center gap-2.5">
              <Compass className="w-6 h-6 text-accent" /> Learning Journey
            </h1>
            <p className="text-text-2 mt-1.5">Your saved study resources. Add material once, then make quizzes from it anytime.</p>
          </div>
          <a href="/add_resource">
            <Button leftIcon={<Plus className="w-4 h-4" />}>Add Resource</Button>
          </a>
        </div>

        {loading ? (
          <div className="grid place-items-center py-24 text-text-3">
            <Loader2 className="w-6 h-6 animate-spin" />
          </div>
        ) : items.length === 0 ? (
          <Card pad="lg" className="text-center py-16">
            <div className="grid place-items-center w-14 h-14 rounded-lg bg-accent/10 text-accent mx-auto mb-4">
              <Library className="w-7 h-7" />
            </div>
            <h3 className="font-display text-lg font-semibold">Your library is empty</h3>
            <p className="text-text-2 mt-1.5 max-w-md mx-auto">
              Save your notes, textbook chapters, or slides here. We index them so you can generate quizzes and revise on demand — no need to re-upload each time.
            </p>
            <a href="/add_resource" className="inline-block mt-5">
              <Button leftIcon={<Plus className="w-4 h-4" />}>Add your first resource</Button>
            </a>
          </Card>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {items.map((k, i) => (
              <motion.div
                key={k.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
              >
                <Card pad="md" className="h-full flex flex-col">
                  <div className="flex items-center justify-between mb-3">
                    <Badge tone="neutral">{k.subject}</Badge>
                    {k.indexed ? (
                      <span className="flex items-center gap-1 text-success text-xs font-medium">
                        <CheckCircle2 className="w-3.5 h-3.5" /> AI Indexed
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-warning text-xs font-medium">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" /> {k.status}
                      </span>
                    )}
                  </div>

                  <h3 className="font-display font-semibold leading-snug line-clamp-2 min-h-[2.6rem]">{k.title}</h3>

                  <div className="flex items-center gap-4 text-text-3 text-xs mt-3 mb-4">
                    <span className="flex items-center gap-1"><Layers className="w-3.5 h-3.5" /> {k.topic_count} topics</span>
                    <span className="flex items-center gap-1"><Clock className="w-3.5 h-3.5" /> ~{k.est_minutes} min</span>
                  </div>

                  <div className="flex-1" />

                  <a href={`/practice?doc=${k.id}`} className="block mb-2">
                    <Button variant="secondary" size="sm" className="w-full" leftIcon={<Dumbbell className="w-3.5 h-3.5" />}>
                      Make Quiz
                    </Button>
                  </a>
                  <button
                    onClick={() => remove(k.id)}
                    disabled={busy === k.id}
                    className="flex items-center justify-center gap-1.5 w-full py-2 rounded-md text-text-3 hover:text-danger hover:bg-danger/8 transition-colors text-xs disabled:opacity-50"
                  >
                    {busy === k.id ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />} Remove
                  </button>
                </Card>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </AppShell>
  );
}
