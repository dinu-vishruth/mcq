import { useState, useEffect } from "react";
import { ArrowRight, BookOpen, RotateCcw, Loader2, Check } from "lucide-react";
import type { DashboardApi } from "@/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { AppShell } from "@/components/AppShell";
import { apiGet } from "@/bootstrap";

export default function Dashboard({ data }: { data: Partial<DashboardApi> & { username: string; error?: string } }) {
  const preloaded = typeof data.total_quizzes === "number" ? data as DashboardApi : null;
  const [d, setD] = useState<DashboardApi | null>(preloaded);
  const [loading, setLoading] = useState(!preloaded);
  useEffect(() => {
    if (preloaded) return;
    apiGet<DashboardApi>("/api/dashboard").then(setD).catch(() => setD(null)).finally(() => setLoading(false));
  }, [preloaded]);
  const learning = d?.learning;
  return (
    <AppShell active="dashboard" username={d?.username || data.username}>
      <div className="max-w-5xl mx-auto px-5 md:px-8 py-10">
        <p className="text-accent-2 text-sm mb-2">Your study desk</p>
        <h1 className="font-display text-3xl font-semibold">What will you understand next?</h1>
        <p className="text-text-2 mt-3 mb-8">Continue with your material, revisit a difficult idea, or see what changed.</p>
        {data.error && <p role="alert" className="mb-5 text-danger">{data.error}</p>}
        {loading ? <Loader2 className="animate-spin" aria-label="Loading progress" /> : !d ? (
          <Card><p role="alert">Your learning progress could not load. Refresh to try again.</p></Card>
        ) : <>
          <Card pad="lg" className="mb-6 bg-gradient-to-br from-accent/10 to-card">
            <BookOpen className="text-accent w-6 h-6 mb-4" />
            <h2 className="font-display text-xl font-semibold">Continue studying</h2>
            <p className="text-text-2 mt-2 mb-5">{learning?.reviewable
              ? `${learning.reviewable} saved questions are available to revisit. Start with earlier mistakes and uncertain answers.`
              : "Make a quiz from your notes or choose a resource from your library."}</p>
            <div className="flex flex-wrap gap-3">
              <a href={learning?.reviewable ? "/review" : "/knowledge"}><Button rightIcon={<ArrowRight className="w-4 h-4" />}>{learning?.reviewable ? "Review saved questions" : "Make a quiz"}</Button></a>
              <a href="/journey"><Button variant="secondary">Open library</Button></a>
            </div>
          </Card>
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <Card pad="lg">
              <RotateCcw className="text-warning w-5 h-5 mb-3" />
              <h2 className="font-display text-lg font-semibold">Topics to revisit</h2>
              {d.weak_topics.length ? <div className="mt-4 space-y-4">{d.weak_topics.slice(0, 4).map(w => (
                <div key={w.topic}><a href="/weak-topics" className="text-accent hover:underline">{w.topic}</a>
                  <p className="text-text-3 text-sm">Missed {w.wrong} of {w.total} recorded answers</p></div>
              ))}</div> : <p className="text-text-2 mt-3">Topic patterns will appear as you practise. You can review saved mistakes even before topics are identified.</p>}
              <p className="text-text-3 text-xs mt-4">Based on recorded answers; this is an estimate of understanding.</p>
              <a href="/review" className="inline-block mt-4 text-accent text-sm">{learning?.to_revisit ?? 0} questions to revisit &rarr;</a>
            </Card>
            <Card pad="lg">
              <Check className="text-success w-5 h-5 mb-3" />
              <h2 className="font-display text-lg font-semibold">Recent improvement</h2>
              <p className="font-display text-3xl mt-4 mb-2">{learning?.corrected ?? 0}</p>
              <p className="text-text-2">Earlier mistakes answered correctly on a later attempt.</p>
              <p className="text-text-3 text-sm mt-3">{learning?.uncertain ?? 0} correct answers were marked unsure or guessing. Revisit them to check your recall.</p>
              <a href="/review" className="inline-block mt-4 text-accent text-sm">See learning evidence &rarr;</a>
            </Card>
          </div>
          <div className="flex items-center justify-between mb-4"><h2 className="font-display text-lg font-semibold">Learning history</h2><a href="/progress" className="text-accent text-sm">View progress</a></div>
          {d.recent.length ? <div className="space-y-3">{d.recent.map((r, i) => (
            <Card key={`${r.session_key}-${i}`} pad="md" className="flex flex-wrap justify-between gap-3">
              <div><span className="font-medium">{r.score} of {r.total} correct</span><p className="text-text-3 text-sm">{r.difficulty || "Mixed practice"} · {(r.submitted_at || "").slice(0, 10)}</p></div>
              <a href="/review" className="text-accent text-sm">Review answers &rarr;</a>
            </Card>
          ))}</div> : <Card><p className="text-text-2">Your first quiz will give you a starting point. Add your material when you are ready.</p></Card>}
        </>}
      </div>
    </AppShell>
  );
}
