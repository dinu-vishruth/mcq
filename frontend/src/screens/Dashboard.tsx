import { useState, useEffect } from "react";
import { BookOpen, FileQuestion, FolderOpen, RotateCcw, Loader2, Check, Plus, Clock } from "lucide-react";
import type { DashboardApi } from "@/types";
import { Card } from "@/components/ui/Card";
import { AppShell } from "@/components/AppShell";
import { apiGet } from "@/bootstrap";

const primaryLink = "inline-flex items-center justify-center gap-2 rounded-md bg-accent px-5 py-3 text-sm font-semibold text-white shadow-glow transition-colors hover:bg-accent-strong";
const secondaryLink = "inline-flex items-center justify-center gap-2 rounded-md border border-line bg-card px-5 py-3 text-sm font-medium transition-colors hover:bg-card-hover";

function StudyNotebook() {
  return (
    <div aria-hidden="true" className="hidden md:flex items-center justify-center relative w-[260px] shrink-0 py-6">
      <div className="absolute w-56 h-56 rounded-full bg-[#e9eddf]/80" />
      <div className="study-notebook relative w-[190px] h-[236px] border border-[#dde2d6] rounded-r-lg rounded-l-sm -rotate-6 pl-7 pr-5 pt-7">
        <span className="absolute inset-y-0 left-4 border-l border-[#e6b9ad]/60" />
        <span className="inline-block bg-[#fffefa] text-[10px] font-semibold tracking-[0.18em] text-text-3 py-1">A NOTE TO SELF</span>
        <p className="font-editorial text-[27px] leading-[1.3] mt-5 bg-[#fffefa] py-2">You do not have<br />to know it all.<br /><span className="text-accent italic">Just begin.</span></p>
        <div className="mt-4 flex items-center gap-2 text-success"><span className="w-5 h-5 rounded-full bg-success/10 grid place-items-center"><Check className="w-3 h-3" /></span><span className="text-[11px] bg-[#fffefa]">One idea at a time</span></div>
      </div>
      <span className="absolute bottom-7 right-3 w-2.5 h-36 rounded-full bg-[#d9ad6a] rotate-[24deg] border-b-[14px] border-text/70 shadow-sm" />
    </div>
  );
}

export default function Dashboard({ data }: { data: Partial<DashboardApi> & { username: string; error?: string } }) {
  const preloaded = typeof data.total_quizzes === "number" ? data as DashboardApi : null;
  const [d, setD] = useState<DashboardApi | null>(preloaded);
  const [loading, setLoading] = useState(!preloaded);
  useEffect(() => {
    if (preloaded) return;
    apiGet<DashboardApi>("/api/dashboard").then(setD).catch(() => setD(null)).finally(() => setLoading(false));
  }, [preloaded]);
  const learning = d?.learning;
  const name = d?.username || data.username;
  const recent = d?.recent || [];
  const topics = d?.weak_topics || [];

  return (
    <AppShell active="dashboard" username={name}>
      <div className="max-w-[1240px] mx-auto px-5 md:px-8 xl:px-12 py-8 md:py-10">
        <header className="mb-7 md:mb-9">
          <p className="text-accent text-xs font-semibold uppercase tracking-[0.15em] mb-2">Your study space</p>
          <h1 className="study-page-title text-3xl md:text-[40px]">Welcome back, {name}.</h1>
          <p className="text-text-2 text-sm mt-3">A fresh page, a little focus. What would you like to learn today?</p>
        </header>
        {data.error && <p role="alert" className="mb-5 text-danger">{data.error}</p>}

        <section aria-labelledby="start-studying" className="relative overflow-hidden rounded-xl border border-line bg-[#f0f2e9] px-6 py-8 md:px-9 md:py-9 mb-7 flex items-center justify-between gap-5">
          <div className="relative z-10 max-w-lg">
            <span className="inline-flex items-center gap-2 text-xs font-medium text-text-2 mb-5"><span className="h-px w-6 bg-text-3" /> A GOOD PLACE TO START</span>
            <h2 id="start-studying" className="study-page-title text-[35px] md:text-[46px] mb-4">Turn your notes into<br className="hidden xl:block" /> something you know.</h2>
            <p className="text-sm text-text-2 max-w-sm leading-relaxed">Bring your notes, pick a topic, and make a quiz that helps the ideas stick.</p>
            <div className="flex flex-wrap gap-3 mt-6">
              <a href="/knowledge" className={primaryLink}><Plus className="w-4 h-4" /> Create a quiz</a>
              <a href="/journey" className={secondaryLink}><FolderOpen className="w-4 h-4" /> Your library</a>
            </div>
          </div>
          <StudyNotebook />
        </section>

        {loading ? <div role="status" className="flex items-center gap-3 py-8 text-text-2"><Loader2 className="w-5 h-5 animate-spin" /> Loading your study space...</div> : !d ? (
          <Card><p role="alert">Your learning progress could not load. Refresh to try again.</p></Card>
        ) : <>
          <dl className="grid grid-cols-1 sm:grid-cols-3 gap-3 md:gap-5 mb-9">
            {[
              { label: "Resources in your library", value: d.knowledge_count ?? 0, icon: BookOpen, tone: "bg-accent/[0.07] text-accent" },
              { label: "Quizzes completed", value: d.total_quizzes, icon: FileQuestion, tone: "bg-success/[0.07] text-success" },
              { label: "Questions to revisit", value: learning?.to_revisit ?? 0, icon: RotateCcw, tone: "bg-warning/[0.07] text-warning" },
            ].map(stat => <div key={stat.label} className="flex items-center gap-4 rounded-lg border border-line bg-card px-5 py-5 shadow-sm">
              <span className={`grid place-items-center w-11 h-11 rounded-md shrink-0 ${stat.tone}`}><stat.icon className="w-5 h-5" /></span>
              <div><dd className="font-display text-2xl font-semibold leading-tight">{stat.value}</dd><dt className="text-xs text-text-2 mt-1">{stat.label}</dt></div>
            </div>)}
          </dl>

          <div className="grid xl:grid-cols-[1.45fr_1fr] gap-7">
            <section aria-labelledby="recent-practice">
              <div className="flex items-center justify-between gap-3 mb-4">
                <h2 id="recent-practice" className="font-display text-base font-semibold">Recent practice</h2>
                <a href="/progress" className="text-accent text-xs font-medium hover:underline">View progress</a>
              </div>
              <Card className="!p-0 overflow-hidden">
                {recent.length ? <div className="divide-y divide-line">{recent.slice(0, 4).map((r, i) => (
                  <div key={`${r.session_key}-${i}`} className="flex items-center gap-4 px-5 py-5">
                    <span className="grid place-items-center w-10 h-11 rounded-sm border border-line bg-bg shrink-0"><FileQuestion className="w-5 h-5 text-text-3" /></span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold">{r.score} of {r.total} correct</p>
                      <p className="text-text-3 text-xs mt-1 capitalize">{r.difficulty || "Mixed practice"}<span className="mx-2">&middot;</span>{(r.submitted_at || "").slice(0, 10)}</p>
                    </div>
                    <a href="/review" className="text-accent text-xs font-medium hover:underline">Review<span className="sr-only"> saved answers</span></a>
                  </div>
                ))}</div> : <div className="p-7 md:p-9 text-center">
                  <span className="inline-grid place-items-center w-12 h-12 rounded-lg bg-card-2 text-text-3 mb-4"><BookOpen className="w-6 h-6" /></span>
                  <h3 className="font-medium text-base mb-2">Your first chapter is waiting.</h3>
                  <p className="text-text-2 text-sm max-w-xs mx-auto leading-relaxed">Start with something you are studying. Your completed quizzes will appear here.</p>
                  <a href="/upload" className="inline-flex items-center gap-2 text-sm font-medium text-accent mt-5 hover:underline"><Plus className="w-4 h-4" /> Add study material</a>
                </div>}
              </Card>
              <div className="flex items-start gap-3 mt-5 px-1 text-text-3 text-xs leading-relaxed"><Clock className="w-4 h-4 shrink-0 mt-0.5" /><p>Study at your own pace. A short quiz is a good way to check what you remember.</p></div>
            </section>

            <section aria-labelledby="a-little-review">
              <h2 id="a-little-review" className="font-display text-base font-semibold mb-4">A little review goes a long way</h2>
              <Card pad="lg">
                <div className="flex items-center gap-3 mb-4"><span className="grid place-items-center w-9 h-9 rounded-full bg-[#f2e9dc] text-warning"><RotateCcw className="w-4 h-4" /></span><h3 className="font-medium text-sm">Pick up where you left off</h3></div>
                <p className="text-text-2 text-sm leading-relaxed mb-5">{learning?.reviewable
                  ? `${learning.to_revisit ?? 0} questions to revisit. Take another look at earlier mistakes and answers you were unsure about.`
                  : "As you practise, this space will help you find the ideas that deserve another look."}</p>
                {topics.length > 0 && <ul className="space-y-3 border-t border-line pt-4 mb-5">{topics.slice(0, 3).map(w => (
                  <li key={w.topic} className="flex items-center justify-between gap-3 text-xs"><a href="/weak-topics" className="text-text-2 hover:text-accent hover:underline">{w.topic}</a><span className="text-text-3 shrink-0">{w.wrong} of {w.total} missed</span></li>
                ))}</ul>}
                <a href={learning?.reviewable ? "/review" : "/knowledge"} className={`${secondaryLink} w-full`}>{learning?.reviewable ? "Review saved questions" : "Make your first quiz"}</a>
                {(learning?.corrected ?? 0) > 0 && <p className="flex items-start gap-2 text-success text-xs mt-4"><Check className="w-4 h-4 shrink-0" />{learning?.corrected} earlier mistakes answered correctly on a later attempt.</p>}
                <p className="text-text-3 text-xs leading-relaxed mt-4">Based on your recorded answers. Every attempt is a chance to understand a little more.</p>
              </Card>
            </section>
          </div>
        </>}
        <footer className="mt-12 pt-5 border-t border-line flex flex-wrap gap-2 justify-between text-xs text-text-3"><span>Examly &middot; Your space to understand.</span><span>Small steps. Lasting understanding.</span></footer>
      </div>
    </AppShell>
  );
}
