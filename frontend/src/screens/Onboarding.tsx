import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  GraduationCap, Briefcase, MessagesSquare, Award, Trophy, Sparkles,
  Dumbbell, RotateCcw, Lightbulb, Layers, ArrowRight, Check,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { csrfToken } from "@/bootstrap";
import type { UserPrefs } from "@/types";

const GOALS = [
  { id: "college", label: "College Exams", icon: GraduationCap },
  { id: "placement", label: "Placement", icon: Briefcase },
  { id: "interview", label: "Interview", icon: MessagesSquare },
  { id: "competitive", label: "Competitive Exams", icon: Trophy },
  { id: "certification", label: "Certification", icon: Award },
  { id: "personal", label: "Personal Learning", icon: Sparkles },
];

const STYLES = [
  { id: "practice", label: "Practice", icon: Dumbbell, desc: "Learn by doing questions" },
  { id: "revision", label: "Revision", icon: RotateCcw, desc: "Quick recaps and notes" },
  { id: "concepts", label: "Concepts", icon: Lightbulb, desc: "Deep understanding first" },
  { id: "mixed", label: "Mixed", icon: Layers, desc: "A balance of everything" },
];

const TIMES = [15, 30, 60, 120];
const timeLabel = (m: number) => (m >= 60 ? `${m / 60} hour${m > 60 ? "s" : ""}` : `${m} mins`);

export function Onboarding({ onDone, initial }: { onDone: () => void; initial: UserPrefs | null }) {
  const [step, setStep] = useState(0);
  const [goal, setGoal] = useState(initial?.goal ?? "");
  const [style, setStyle] = useState(initial?.style ?? "");
  const [minutes, setMinutes] = useState(initial?.daily_minutes ?? 30);
  const [saving, setSaving] = useState(false);

  async function finish() {
    setSaving(true);
    try {
      await fetch("/api/prefs", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken() },
        body: JSON.stringify({ goal, style, daily_minutes: minutes }),
      });
    } catch {
      /* best-effort; onboarding shouldn't block the app */
    }
    onDone();
  }

  const canNext = step === 0 ? !!goal : step === 1 ? !!style : true;

  return (
    <div className="fixed inset-0 z-50 bg-bg/95 backdrop-blur-sm grid place-items-center p-5 app-glow">
      <motion.div
        initial={{ opacity: 0, scale: 0.97, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-lg bg-card border border-white/[0.11] rounded-xl p-7 shadow-lg"
      >
        <div className="flex gap-1.5 mb-6">
          {[0, 1, 2].map((i) => (
            <div key={i} className={`h-1 flex-1 rounded-full transition-colors ${i <= step ? "bg-accent" : "bg-white/10"}`} />
          ))}
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, x: 16 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -16 }}
            transition={{ duration: 0.25 }}
          >
            {step === 0 && (
              <>
                <h2 className="text-2xl mb-1">Welcome! 👋</h2>
                <p className="text-text-2 mb-6">What are you preparing for?</p>
                <div className="grid grid-cols-2 gap-3">
                  {GOALS.map((g) => {
                    const Icon = g.icon;
                    const on = goal === g.id;
                    return (
                      <button
                        key={g.id}
                        onClick={() => setGoal(g.id)}
                        className={`flex items-center gap-3 p-3.5 rounded-md border text-left transition-all ${
                          on ? "border-accent bg-accent/10 text-text" : "border-white/[0.08] text-text-2 hover:border-white/20"
                        }`}
                      >
                        <Icon className="w-5 h-5 shrink-0" />
                        <span className="text-sm font-medium">{g.label}</span>
                      </button>
                    );
                  })}
                </div>
              </>
            )}

            {step === 1 && (
              <>
                <h2 className="text-2xl mb-1">Learning style</h2>
                <p className="text-text-2 mb-6">How do you like to learn?</p>
                <div className="grid grid-cols-2 gap-3">
                  {STYLES.map((s) => {
                    const Icon = s.icon;
                    const on = style === s.id;
                    return (
                      <button
                        key={s.id}
                        onClick={() => setStyle(s.id)}
                        className={`p-4 rounded-md border text-left transition-all ${
                          on ? "border-accent bg-accent/10" : "border-white/[0.08] hover:border-white/20"
                        }`}
                      >
                        <Icon className={`w-5 h-5 mb-2 ${on ? "text-accent" : "text-text-2"}`} />
                        <div className="text-sm font-medium">{s.label}</div>
                        <div className="text-xs text-text-3 mt-0.5">{s.desc}</div>
                      </button>
                    );
                  })}
                </div>
              </>
            )}

            {step === 2 && (
              <>
                <h2 className="text-2xl mb-1">Daily study time</h2>
                <p className="text-text-2 mb-6">How much can you commit each day?</p>
                <div className="grid grid-cols-2 gap-3">
                  {TIMES.map((m) => {
                    const on = minutes === m;
                    return (
                      <button
                        key={m}
                        onClick={() => setMinutes(m)}
                        className={`p-5 rounded-md border text-center transition-all ${
                          on ? "border-accent bg-accent/10 text-text" : "border-white/[0.08] text-text-2 hover:border-white/20"
                        }`}
                      >
                        <div className="text-xl font-display font-bold">{timeLabel(m)}</div>
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </motion.div>
        </AnimatePresence>

        <div className="flex items-center justify-between mt-7">
          <button
            onClick={() => (step === 0 ? onDone() : setStep(step - 1))}
            className="text-text-3 text-sm hover:text-text-2 transition-colors"
          >
            {step === 0 ? "Skip for now" : "Back"}
          </button>
          {step < 2 ? (
            <Button disabled={!canNext} rightIcon={<ArrowRight className="w-4 h-4" />} onClick={() => setStep(step + 1)}>
              Next
            </Button>
          ) : (
            <Button disabled={saving} leftIcon={<Check className="w-4 h-4" />} onClick={finish}>
              {saving ? "Saving…" : "Finish"}
            </Button>
          )}
        </div>
      </motion.div>
    </div>
  );
}
