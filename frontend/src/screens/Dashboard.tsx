import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Play, Sparkles, Target, Clock, Flame, Zap, BookOpen, Dumbbell,
  TrendingUp, Brain, ArrowRight, RefreshCw, AlertTriangle, Bell, Loader2, ChevronRight,
} from "lucide-react";
import type { DashboardApi } from "@/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { StatCard } from "@/components/ui/StatCard";
import { TrendChart } from "@/components/charts/TrendChart";
import { Onboarding } from "./Onboarding";
import { AppShell } from "@/components/AppShell";
import { apiGet } from "@/bootstrap";

function greeting(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

const REC_ICON: Record<string, typeof Play> = {
  revision: RefreshCw, practice: Dumbbell, start: Sparkles, interview: Brain,
};
const REC_HREF: Record<string, string> = {
  revision: "/weak-topics", practice: "/practice", start: "/knowledge", interview: "/practice",
};

// The server embeds the complete dashboard payload in #root's data-bootstrap,
// so this screen renders on first paint. It used to mount empty, show a spinner,
// and fetch /api/dashboard — a second round-trip for data the page already had,
// which on a cold serverless start meant staring at the spinner for seconds.
// Only fall back to fetching if the payload is absent (a route that rendered the
// template without it).
export default function Dashboard({ data }: { data: Partial<DashboardApi> & { username: string } }) {
  const preloaded = typeof data.total_quizzes === "number" ? (data as DashboardApi) : null;

  const [d, setD] = useState<DashboardApi | null>(preloaded);
  const [loading, setLoading] = useState(preloaded === null);
  const [showOnboarding, setShowOnboarding] = useState(Boolean(preloaded?.needs_onboarding));

  useEffect(() => {
    if (preloaded) return;
    apiGet<DashboardApi>("/api/dashboard")
      .then((res) => {
        setD(res);
        if (res.needs_onboarding) setShowOnboarding(true);
      })
      .catch(() => setD(null))
      .finally(() => setLoading(false));
  }, [preloaded]);

  const username = d?.username || data.username;

  if (loading) {
    return (
      <AppShell active="dashboard" username={username}>
        <div className="grid place-items-center min-h-[60vh] text-text-3"><Loader2 className="w-6 h-6 animate-spin" /></div>
      </AppShell>
    );
  }

  const hasHistory = (d?.total_quizzes ?? 0) > 0;
  const topWeak = d?.weak_topics?.[0];
  const resume = d?.recent?.[0];

  return (
    <AppShell active="dashboard" username={username} streak={d?.streak} xp={d?.xp}>
      {showOnboarding && <Onboarding onDone={() => setShowOnboarding(false)} initial={null} />}

      <div className="max-w-6xl mx-auto px-5 md:px-8 py-8 md:py-10">
        {/* Hero — what should I study today? */}
        <motion.header initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="mb-8">
          <p className="text-accent-2 text-sm font-medium mb-2 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4" /> Your exam coach
          </p>
          <h1 className="text-[clamp(1.9rem,3.2vw,2.5rem)] font-bold leading-tight">
            {greeting()}, {username}
          </h1>
          <p className="text-text-2 mt-2 max-w-xl">
            {topWeak
              ? `Today, focus on ${topWeak.topic} — it's your biggest opportunity to improve.`
              : hasHistory
              ? "You're on track. A short session today keeps your streak and recall sharp."
              : "Let's build your baseline. Add a study source and I'll turn it into practice."}
          </p>
        </motion.header>

        {/* Today's plan: primary CTA + goal + streak */}
        <div className="grid lg:grid-cols-3 gap-4 mb-8">
          <Card pad="lg" className="lg:col-span-2 bg-gradient-to-br from-accent/[0.08] to-card">
            <div className="flex items-center gap-2 text-accent-2 text-sm font-medium mb-2">
              <Target className="w-4 h-4" /> Today's study goal
            </div>
            <h2 className="font-display text-xl font-semibold mb-1">
              {d?.daily_minutes ?? 30} minutes · {topWeak ? topWeak.topic : "mixed practice"}
            </h2>
            <p className="text-text-2 text-sm mb-5">
              {topWeak
                ? "A focused revision plus a short practice set will move the needle fastest."
                : "Generate a quick set from your knowledge to keep momentum."}
            </p>
            <div className="flex flex-wrap gap-3">
              {resume ? (
                <a href="/practice"><Button leftIcon={<Play className="w-4 h-4" />}>Start today's session</Button></a>
              ) : (
                <a href={hasHistory ? "/practice" : "/knowledge"}>
                  <Button leftIcon={<Play className="w-4 h-4" />}>{hasHistory ? "Start practicing" : "Add knowledge"}</Button>
                </a>
              )}
              {topWeak && (
                <a href="/weak-topics"><Button variant="secondary" leftIcon={<RefreshCw className="w-4 h-4" />}>Revise weak topics</Button></a>
              )}
            </div>
          </Card>

          <Card pad="lg" className="flex flex-col justify-center">
            <div className="flex items-center justify-between mb-3">
              <span className="flex items-center gap-2 text-sm font-medium"><Flame className="w-4 h-4 text-warning" /> Streak</span>
              <Badge tone="warning">{d?.streak ?? 0} day{(d?.streak ?? 0) !== 1 ? "s" : ""}</Badge>
            </div>
            <div className="flex gap-1.5 mb-4">
              {["M", "T", "W", "T", "F", "S", "S"].map((lbl, i) => (
                <div key={i} className={`flex-1 aspect-square grid place-items-center rounded-sm text-xs font-medium ${
                  i < (d?.streak ?? 0) ? "bg-warning/20 text-warning" : "bg-white/[0.04] text-text-3"}`}>{lbl}</div>
              ))}
            </div>
            <div className="flex items-center gap-2.5 pt-3 border-t border-white/[0.06]">
              <span className="grid place-items-center w-9 h-9 rounded-md bg-violet/10 text-violet"><Zap className="w-4 h-4" /></span>
              <div>
                <div className="font-semibold leading-none">{d?.xp ?? 0} XP</div>
                <div className="text-text-3 text-xs mt-0.5">Level {d?.level ?? 1}</div>
              </div>
            </div>
          </Card>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard index={0} value={d?.total_quizzes ?? 0} label="Sessions" icon={<BookOpen />} tone="accent" />
          <StatCard index={1} value={`${d?.avg_score ?? 0}%`} label="Avg Score" icon={<Target />} tone="emerald" />
          <StatCard index={2} value={d?.knowledge_count ?? 0} label="Saved Resources" icon={<Brain />} tone="violet" />
          <StatCard index={3}
            value={(d?.total_time ?? 0) >= 60 ? `${Math.floor((d?.total_time ?? 0) / 60)}m` : `${d?.total_time ?? 0}s`}
            label="Study Time" icon={<Clock />} tone="amber" />
        </div>

        {/* Recommendations — proactive coach */}
        {(d?.recommendations?.length ?? 0) > 0 && (
          <div className="mb-8">
            <h2 className="font-display font-semibold mb-3 flex items-center gap-2"><Sparkles className="w-4 h-4 text-accent" /> Recommended for today</h2>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {d!.recommendations.map((r, i) => {
                const Icon = REC_ICON[r.kind] ?? Play;
                return (
                  <motion.a key={i} href={REC_HREF[r.kind] ?? "/practice"}
                    initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
                    <Card hover pad="md" className="h-full">
                      <span className="grid place-items-center w-10 h-10 rounded-md bg-accent/10 text-accent mb-3"><Icon className="w-5 h-5" /></span>
                      <div className="font-medium">{r.title}</div>
                      <p className="text-text-3 text-xs mt-1 mb-3 leading-relaxed">{r.reason}</p>
                      <span className="text-accent text-sm font-medium flex items-center gap-1">{r.cta} <ChevronRight className="w-3.5 h-3.5" /></span>
                    </Card>
                  </motion.a>
                );
              })}
            </div>
          </div>
        )}

        {/* Trend + weak topics */}
        {hasHistory && (
          <div className="grid lg:grid-cols-2 gap-4 mb-8">
            <Card>
              <h2 className="flex items-center gap-2 text-lg"><TrendingUp className="w-5 h-5 text-accent" /> Weekly performance</h2>
              <p className="text-text-3 text-sm mb-4 -mt-2">Your scores over recent sessions (%)</p>
              <TrendChart dates={(d?.chart ?? []).map((c) => c.date)} scores={(d?.chart ?? []).map((c) => c.score)} />
            </Card>
            <Card>
              <div className="flex items-center justify-between">
                <h2 className="flex items-center gap-2 text-lg"><AlertTriangle className="w-5 h-5 text-warning" /> Weak topics</h2>
                <a href="/weak-topics" className="text-accent text-sm hover:underline">View all</a>
              </div>
              <p className="text-text-3 text-sm mb-4 -mt-2">What to review next</p>
              {(d?.weak_topics?.length ?? 0) ? (
                <div className="space-y-4">
                  {d!.weak_topics.slice(0, 4).map((w) => (
                    <div key={w.topic}>
                      <div className="flex items-center justify-between mb-1.5">
                        <strong className="text-sm">{w.topic}</strong>
                        <Badge tone="danger">{w.pct}%</Badge>
                      </div>
                      <ProgressBar pct={w.pct} tone="danger" />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-text-3 text-sm">No weak areas detected yet. Keep practicing.</p>
              )}
            </Card>
          </div>
        )}

        {/* Upcoming revision reminders */}
        {topWeak && (
          <Card pad="md" className="mb-8 flex items-start gap-4 bg-gradient-to-br from-violet/[0.08] to-card">
            <span className="grid place-items-center w-11 h-11 rounded-md bg-violet/15 text-violet shrink-0"><Bell className="w-5 h-5" /></span>
            <div className="flex-1">
              <h2 className="font-display font-semibold mb-1">Revision reminder</h2>
              <p className="text-text-2 text-sm mb-3">
                {topWeak.topic} is due for a review — spaced repetition works best before you forget.
              </p>
              <a href="/weak-topics"><Button variant="secondary" size="sm" rightIcon={<ArrowRight className="w-4 h-4" />}>Revise now</Button></a>
            </div>
          </Card>
        )}

        {/* Resume card */}
        <h2 className="font-display font-semibold mb-3">Recent study sessions</h2>
        {(d?.recent?.length ?? 0) ? (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {d!.recent.map((h, i) => (
              <motion.div key={`${h.session_key}-${i}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                <Card hover pad="md">
                  <div className="flex items-center justify-between mb-3">
                    <Badge tone="neutral">{h.difficulty || "mixed"}</Badge>
                    <Badge tone={h.pct >= 70 ? "success" : "danger"}>{h.pct}%</Badge>
                  </div>
                  <div className="text-sm text-text-2">{h.score} / {h.total} correct</div>
                  <div className="text-text-3 text-xs mt-2 flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5" /> {(h.submitted_at || "").slice(0, 16).replace("T", " ")}
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        ) : (
          <Card className="text-center py-12">
            <span className="grid place-items-center w-14 h-14 rounded-full bg-white/5 text-text-2 mx-auto mb-4"><BookOpen className="w-7 h-7" /></span>
            <h2 className="text-lg mb-1">No sessions yet</h2>
            <p className="text-text-3 text-sm mb-5">Make your first quiz from a document or a saved resource.</p>
            <a href="/knowledge"><Button leftIcon={<Sparkles className="w-4 h-4" />}>Make Quiz &amp; Test</Button></a>
          </Card>
        )}
      </div>
    </AppShell>
  );
}
