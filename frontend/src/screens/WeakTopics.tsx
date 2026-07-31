import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle, Loader2, RotateCcw, CheckCircle2, TrendingDown,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { apiGet, apiSend, csrfToken } from "@/bootstrap";

interface WeakItem {
  topic: string; wrong: number; total: number; pct: number;
  severity: "high" | "medium" | "low"; reviewable: number;
}

const SEV: Record<string, { tone: "danger" | "warning" | "success"; label: string }> = {
  high: { tone: "danger", label: "Needs work" },
  medium: { tone: "warning", label: "Shaky" },
  low: { tone: "success", label: "Almost there" },
};

export default function WeakTopics({ data }: { data: { username: string } }) {
  const [items, setItems] = useState<WeakItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState<string | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiGet<{ items: WeakItem[] }>("/api/weak-topics")
      .then((d) => setItems(d.items))
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, []);

  async function review(topic: string) {
    setStarting(topic);
    setError("");
    try {
      const res = await apiSend<{ session_key: string }>("/api/weak-topics/review", "POST", { topic });
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
      setError(e instanceof Error ? e.message : "Could not start review. Try again.");
      setStarting(null);
    }
  }

  return (
    <AppShell active="weak-topics" username={data.username}>
      <div className="max-w-4xl mx-auto px-5 lg:px-8 py-8">
        <h1 className="font-display text-2xl font-semibold flex items-center gap-2.5">
          <AlertTriangle className="w-6 h-6 text-warning" /> Weak Topics
        </h1>
        <p className="text-text-2 mt-1.5 mb-8">
          Your coach tracks concepts you miss and points you to the fastest way to fix them.
        </p>

        {error && <div className="mb-4 text-danger text-sm bg-danger/8 border border-danger/20 rounded-md px-4 py-3">{error}</div>}

        {loading ? (
          <div className="grid place-items-center py-24 text-text-3"><Loader2 className="w-6 h-6 animate-spin" /></div>
        ) : items.length === 0 ? (
          <Card pad="lg" className="text-center py-16">
            <div className="grid place-items-center w-14 h-14 rounded-lg bg-success/10 text-success mx-auto mb-4">
              <CheckCircle2 className="w-7 h-7" />
            </div>
            <h3 className="font-display text-lg font-semibold">No weak spots detected yet</h3>
            <p className="text-text-2 mt-1.5 max-w-md mx-auto">
              Complete a few practice sets and your coach will surface the topics worth revisiting here.
            </p>
            <a href="/practice" className="inline-block mt-5"><Button>Start practicing</Button></a>
          </Card>
        ) : (
          <>
            <div className="flex items-center gap-2 text-text-2 text-sm mb-4">
              <TrendingDown className="w-4 h-4 text-danger" />
              {items.length} topic{items.length > 1 ? "s" : ""} to strengthen, weakest first.
            </div>
            <div className="space-y-3">
              {items.map((w, i) => {
                const sev = SEV[w.severity] ?? SEV.medium;
                return (
                  <motion.div key={w.topic} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                    <Card pad="md">
                      <div className="flex items-start justify-between gap-4 mb-3">
                        <div className="min-w-0">
                          <h3 className="font-display font-semibold truncate">{w.topic}</h3>
                          <p className="text-text-3 text-xs mt-0.5">
                            Missed {w.wrong} of {w.total} · {w.pct}% miss rate
                          </p>
                        </div>
                        <span className={`shrink-0 text-xs font-medium px-2.5 py-1 rounded-full ${
                          sev.tone === "danger" ? "bg-danger/12 text-danger" :
                          sev.tone === "warning" ? "bg-warning/12 text-warning" : "bg-success/12 text-success"}`}>
                          {sev.label}
                        </span>
                      </div>

                      <ProgressBar pct={w.pct} tone={sev.tone === "success" ? "success" : "danger"} />

                      <div className="flex flex-wrap items-center gap-2 mt-4">
                        {w.reviewable > 0 ? (
                          <Button
                            size="sm"
                            onClick={() => review(w.topic)}
                            disabled={starting === w.topic}
                            leftIcon={starting === w.topic
                              ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                              : <RotateCcw className="w-3.5 h-3.5" />}
                          >
                            {starting === w.topic
                              ? "Starting…"
                              : `Review ${w.reviewable} missed question${w.reviewable > 1 ? "s" : ""}`}
                          </Button>
                        ) : (
                          <span className="text-text-3 text-xs">
                            Original questions no longer available to review.
                          </span>
                        )}
                      </div>
                    </Card>
                  </motion.div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </AppShell>
  );
}
