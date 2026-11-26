import { motion } from "framer-motion";
import { Target, Clock, BookOpen, Check, TrendingUp } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";

interface WeeklyPoint { label: string; avg: number; count: number; }
interface WeakTopic { topic: string; wrong: number; total: number; pct: number; }
export interface ProgressData {
  username: string;
  total_quizzes: number;
  accuracy: number;
  total_time: number;
  total_correct: number;
  total_answered: number;
  heatmap: Record<string, number>;
  weekly: WeeklyPoint[];
  weak_topics: WeakTopic[];
}

// Build ~17 weeks (a GitHub-style grid) ending today, using local date math only.
function buildHeatmap(counts: Record<string, number>) {
  const weeks = 17;
  const today = new Date();
  const cells: { date: string; count: number }[] = [];
  const start = new Date(today);
  start.setDate(start.getDate() - (weeks * 7 - 1));
  for (let i = 0; i < weeks * 7; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    const key = d.toISOString().slice(0, 10);
    cells.push({ date: key, count: counts[key] ?? 0 });
  }
  return cells;
}

function level(count: number): string {
  if (count === 0) return "bg-inset";
  if (count === 1) return "bg-accent/30";
  if (count === 2) return "bg-accent/55";
  if (count <= 4) return "bg-accent/75";
  return "bg-accent";
}

export default function Progress({ data }: { data: ProgressData }) {
  const cells = buildHeatmap(data.heatmap);
  const maxWeekly = Math.max(1, ...data.weekly.map((w) => w.avg));
  const activeDays = Object.keys(data.heatmap).length;

  return (
    <AppShell active="progress" username={data.username}>
      <div className="max-w-5xl mx-auto px-5 md:px-8 py-8 md:py-10">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <p className="text-accent-2 text-sm font-medium mb-1 flex items-center gap-1.5"><TrendingUp className="w-4 h-4" /> Your journey</p>
          <h1 className="text-[clamp(1.7rem,3vw,2.3rem)] font-bold mb-8">Learning Progress</h1>
        </motion.div>

        {/* Top stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {[
            { v: `${data.total_quizzes}`, l: "Practice sessions", icon: BookOpen, tone: "text-accent bg-accent/10" },
            { v: `${data.total_correct}`, l: "Correct answers", icon: Check, tone: "text-success bg-success/10" },
            { v: `${data.accuracy}%`, l: "Accuracy", icon: Target, tone: "text-success bg-success/10" },
            { v: data.total_time >= 60 ? `${Math.floor(data.total_time / 60)}m` : `${data.total_time}s`, l: "Study Time", icon: Clock, tone: "text-accent bg-accent/10" },
          ].map((s, i) => {
            const Icon = s.icon;
            return (
              <Card key={i} hover>
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-2xl font-display font-bold">{s.v}</div>
                    <div className="text-text-3 text-sm mt-1">{s.l}</div>
                  </div>
                  <span className={`grid place-items-center w-10 h-10 rounded-sm ${s.tone}`}><Icon className="w-5 h-5" /></span>
                </div>
              </Card>
            );
          })}
        </div>

        {/* Contribution heatmap */}
        <Card className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg mb-0">Learning Activity</h2>
            <span className="text-text-3 text-sm">{activeDays} active day{activeDays !== 1 ? "s" : ""}</span>
          </div>
          <div className="grid grid-flow-col grid-rows-7 gap-1 overflow-x-auto pb-1" style={{ gridAutoColumns: "min-content" }}>
            {cells.map((c, i) => (
              <div
                key={i}
                title={`${c.date}: ${c.count} quiz${c.count !== 1 ? "zes" : ""}`}
                className={`w-3 h-3 rounded-[3px] ${level(c.count)}`}
              />
            ))}
          </div>
          <div className="flex items-center gap-2 mt-3 text-text-3 text-xs">
            <span>Less</span>
            <span className="w-3 h-3 rounded-[3px] bg-inset" />
            <span className="w-3 h-3 rounded-[3px] bg-accent/30" />
            <span className="w-3 h-3 rounded-[3px] bg-accent/55" />
            <span className="w-3 h-3 rounded-[3px] bg-accent/75" />
            <span className="w-3 h-3 rounded-[3px] bg-accent" />
            <span>More</span>
          </div>
        </Card>

        <div className="grid lg:grid-cols-2 gap-4 mb-8">
          {/* Monthly performance */}
          <Card>
            <h2 className="text-lg">Performance Over Time</h2>
            <p className="text-text-3 text-sm mb-4 -mt-2">Average score per month (%)</p>
            {data.weekly.length ? (
              <div className="flex items-end gap-2 h-40">
                {data.weekly.map((w, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center gap-1.5">
                    <motion.div
                      className="w-full rounded-t bg-accent/60"
                      initial={{ height: 0 }}
                      animate={{ height: `${(w.avg / maxWeekly) * 100}%` }}
                      transition={{ delay: i * 0.05, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                      style={{ minHeight: 2 }}
                      title={`${w.label}: ${w.avg}%`}
                    />
                    <span className="text-text-3 text-[0.65rem]">{w.label.slice(5)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-text-3 text-sm">Complete quizzes to see your trend.</p>
            )}
          </Card>

          {/* Topic evidence */}
          <Card>
            <h2 className="text-lg">Topic evidence</h2>
            <p className="text-text-3 text-sm mb-4 -mt-2">Estimates from recorded answers; question sets may differ.</p>
            {data.weak_topics.length ? (
              <div className="space-y-3">
                {data.weak_topics.slice(0, 6).map((w) => {
                  const accuracy = 100 - w.pct;
                  return (
                    <div key={w.topic}>
                      <div className="flex items-center justify-between mb-1.5">
                        <strong className="text-sm">{w.topic}</strong>
                        <span className="text-text-3 text-xs">{accuracy}% correct across {w.total} answers</span>
                      </div>
                      <ProgressBar pct={accuracy} tone={accuracy >= 60 ? "success" : "danger"} />
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-text-3 text-sm">No weak topics tracked yet. Keep practicing.</p>
            )}
          </Card>
        </div>

        <Card><h2 className="text-lg">What changed in your understanding?</h2><p className="text-text-2 mt-2 mb-3">See earlier mistakes you have corrected and questions worth another attempt.</p><a href="/review" className="text-accent">Review learning evidence &rarr;</a></Card>
      </div>
    </AppShell>
  );
}
