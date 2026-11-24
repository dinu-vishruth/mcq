import { useEffect, useState } from "react";
import { RotateCcw, Loader2, Check } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { apiGet, apiSend, csrfToken } from "@/bootstrap";
import type { LearningSummary } from "@/types";

interface ReviewItem {
  question: string; selected: string; correct: string; state: string;
  last_seen: string; attempts: number;
  document_id: number | null; document_title: string | null;
}
interface ReviewData { summary: LearningSummary; items: ReviewItem[]; improvements: ReviewItem[] }
const guidance: Record<string, string> = {
  misconception: "You felt confident, but the answer was incorrect. Check for a possible misconception.",
  revisit: "Read the explanation or source material, then try this idea again.",
  uncertain: "You answered correctly while unsure or guessing. Revisit this to check your recall.",
};
export default function ReviewMistakes({ data }: { data: { username: string } }) {
  const [d, setD] = useState<ReviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  function load() {
    setLoading(true); setError("");
    apiGet<ReviewData>("/api/review").then(setD)
      .catch(() => setError("Your saved answers could not load. Please try again."))
      .finally(() => setLoading(false));
  }
  useEffect(load, []);
  async function start() {
    setStarting(true); setError("");
    try {
      const result = await apiSend<{ session_key: string }>("/api/review/start", "POST", {});
      const form = document.createElement("form");
      form.method = "POST"; form.action = "/student_login";
      for (const [name, value] of Object.entries({ csrf_token: csrfToken(), session_key: result.session_key })) {
        const input = document.createElement("input");
        input.type = "hidden"; input.name = name; input.value = value; form.appendChild(input);
      }
      document.body.appendChild(form); form.submit();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start review. Please retry.");
      setStarting(false);
    }
  }
  return <AppShell active="review" username={data.username}>
    <div className="max-w-4xl mx-auto px-5 lg:px-8 py-10">
      <p className="text-accent-2 text-sm mb-2">Learning evidence</p>
      <h1 className="font-display text-3xl font-semibold">Review mistakes</h1>
      <p className="text-text-2 mt-3 mb-8">Understand what went wrong and check what changed on your next attempt.</p>
      {error && <div role="alert" className="mb-5 text-danger">{error} {!d && <button onClick={load} className="underline">Retry</button>}</div>}
      {loading ? <Loader2 className="animate-spin" aria-label="Loading saved answers" /> : d && <>
        <div className="grid sm:grid-cols-3 gap-4 mb-6">
          {[{ value: d.summary.to_revisit, label: "Questions to revisit" }, { value: d.summary.uncertain, label: "Correct with uncertainty" }, { value: d.summary.corrected, label: "Earlier mistakes corrected" }].map(s => (
            <Card key={s.label}><div className="font-display text-3xl">{s.value}</div><p className="text-text-2 text-sm mt-2">{s.label}</p></Card>
          ))}
        </div>
        <p className="text-text-3 text-sm mb-5">Based on your latest answer to each saved question. A later correct answer is progress, rather than proof of mastery.</p>
        {d.summary.reviewable > 0 && <Button onClick={start} disabled={starting} leftIcon={starting ? <Loader2 className="w-4 h-4 animate-spin" /> : <RotateCcw className="w-4 h-4" />}>
          {starting ? "Opening review…" : `Revisit ${d.summary.reviewable} saved questions`}
        </Button>}
        <h2 className="font-display text-xl mt-8 mb-4">Worth another look</h2>
        {d.items.length ? <div className="space-y-4">{d.items.map((q, i) => <Card key={i}>
          <p className="text-accent-2 text-sm mb-2">{q.state === "misconception" ? "Check your reasoning" : q.state === "uncertain" ? "Build confidence" : "Revisit this idea"}</p>
          <h3 className="font-medium mb-3">{q.question}</h3>
          <p className="text-text-2 text-sm">Your answer: {q.selected || "Not answered"}</p>
          <details className="mt-3"><summary className="cursor-pointer text-accent text-sm">Compare with the correct answer</summary><p className="text-text-2 mt-2">{q.correct}</p></details>
          <p className="text-text-2 text-sm mt-3">{guidance[q.state]}</p>
          <p className="text-text-3 text-xs mt-3">Last attempted {(q.last_seen || "").slice(0, 10)} · {q.attempts} attempt{q.attempts === 1 ? "" : "s"}</p>
          {q.document_id && <a href={`/practice?doc=${q.document_id}`} className="inline-block mt-3 text-accent text-sm">Practise from {q.document_title || "the original resource"} &rarr;</a>}
        </Card>)}</div> : <Card><p className="text-text-2">No saved mistakes or uncertain answers to revisit. Make a quiz to explore another part of your material.</p><a href="/knowledge" className="inline-block mt-3 text-accent">Make a quiz &rarr;</a></Card>}
        {d.items.length > 0 && !d.summary.reviewable && <p className="text-text-3 text-sm mt-4">The original quiz options are no longer available. Use your library to make a fresh practice set.</p>}
        {d.summary.to_revisit + d.summary.uncertain > d.items.length && <p className="text-text-3 text-sm mt-4">Showing the 50 most recent questions to revisit.</p>}
        <h2 className="font-display text-xl mt-8 mb-4">What improved</h2>
        {d.improvements.length ? <div className="space-y-3">{d.improvements.map((q, i) => <Card key={i}>
          <p className="text-success text-sm flex gap-2 items-center mb-2"><Check className="w-4 h-4" /> Earlier mistake answered correctly</p><p>{q.question}</p>
          <p className="text-text-3 text-sm mt-2">{q.state === "uncertain" ? "Still marked unsure or guessing; revisit to check recall." : "Check your understanding again later with a different question."}</p>
        </Card>)}</div> : <Card><p className="text-text-2">When you answer an earlier mistake correctly on a later attempt, it will appear here.</p></Card>}
      </>}
    </div>
  </AppShell>;
}
