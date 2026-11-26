import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { User, Sparkles, Download, Trash2, Check, Palette } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { apiSend, csrfToken } from "@/bootstrap";

export interface SettingsData {
  username: string;
  email: string;
  email_locked?: boolean;
  password_enabled?: boolean;
  is_admin?: boolean;
  password_upgrade_required?: boolean;
  password_min_length?: number;
  password_max_length?: number;
  password_help?: string;
  error?: string;
  flash?: string;
}

const GOALS = ["college", "placement", "interview", "competitive", "certification", "personal"];
const STYLES = ["practice", "revision", "concepts", "mixed"];
const TIMES = [15, 30, 60, 120];
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const timeLabel = (m: number) => (m >= 60 ? `${m / 60}h` : `${m}m`);

export default function Settings({ data }: { data: SettingsData }) {
  const [goal, setGoal] = useState("");
  const [style, setStyle] = useState("");
  const [minutes, setMinutes] = useState(30);
  const [prefsSaved, setPrefsSaved] = useState(false);
  const [savingPrefs, setSavingPrefs] = useState(false);
  const [prefsError, setPrefsError] = useState("");

  // Prefs come from the additive /api/prefs endpoint so the
  // existing /profile route in auth.py stays untouched.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/prefs", { headers: { Accept: "application/json" } })
      .then((r) => (r.ok ? r.json() : null))
      .then((p) => {
        if (cancelled || !p?.prefs) return;
        setGoal(p.prefs.goal ?? "");
        setStyle(p.prefs.style ?? "");
        setMinutes(p.prefs.daily_minutes ?? 30);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  async function savePrefs() {
    setSavingPrefs(true);
    setPrefsSaved(false);
    setPrefsError("");
    try {
      await apiSend("/api/prefs", "POST", { goal, style, daily_minutes: minutes });
      setPrefsSaved(true);
      setTimeout(() => setPrefsSaved(false), 2500);
    } catch (err) {
      setPrefsError(err instanceof Error ? err.message : "Could not save preferences. Please retry.");
    } finally {
      setSavingPrefs(false);
    }
  }

  return (
    <AppShell active="settings" username={data.username}>
      <div className="max-w-2xl mx-auto px-5 md:px-8 py-8 md:py-10">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <p className="text-accent-2 text-sm font-medium mb-1">Account</p>
          <h1 className="text-[clamp(1.7rem,3vw,2.3rem)] font-bold mb-8">Settings</h1>
        </motion.div>

        {data.flash && (
          <div className="mb-4 px-4 py-3 rounded-md bg-success/10 border border-success/20 text-success text-sm flex items-center gap-2">
            <Check className="w-4 h-4" /> {data.flash}
          </div>
        )}
        {data.error && (
          <div className="mb-4 px-4 py-3 rounded-md bg-danger/10 border border-danger/20 text-danger text-sm">{data.error}</div>
        )}

        {data.is_admin && <Card className="mb-5 border-accent/20"><h2 className="font-display text-lg">Your admin workspace</h2><p className="text-sm text-text-2 my-3">Manage application access, accounts, and activity.</p><a href="/admin" className="text-accent font-medium text-sm hover:underline">Open admin dashboard</a></Card>}
        {data.password_upgrade_required && <p role="alert" className="mb-5 p-4 rounded-md border border-warning/30 bg-warning/5 text-warning text-sm">Your current password needs an upgrade. Set a unique password of at least {data.password_min_length ?? 8} characters before continuing.</p>}

        {/* Account */}
        <Card className="mb-5">
          <h2 className="flex items-center gap-2 text-lg"><User className="w-5 h-5 text-accent" /> Account Details</h2>
          <form action="/profile" method="post" className="mt-4 space-y-4">
            <input type="hidden" name="csrf_token" value={csrfToken()} />
            <div>
              <label className="text-text-2 text-sm block mb-1.5">Username</label>
              <input name="username" defaultValue={data.username} required
                className="w-full h-11 px-3 rounded-md bg-inset border border-line text-text focus:outline-none focus:border-accent/50" />
              {data.username.normalize("NFKC").includes("@") && (
                <p className="mt-1.5 text-xs text-text-3">Choose a username without an @ sign.</p>
              )}
            </div>
            <div>
              <label className="text-text-2 text-sm block mb-1.5">Email</label>
              <input type="email" name={data.email_locked ? undefined : "email"} defaultValue={data.email}
                readOnly={data.email_locked} placeholder={data.email_locked ? "No email from your sign-in provider" : "name@example.com"}
                className="w-full h-11 px-3 rounded-md bg-inset border border-line text-text focus:outline-none focus:border-accent/50" />
              <p className="mt-1.5 text-xs text-text-3">
                {data.email_locked ? "Managed by your sign-in provider." : "Optional contact email. Sign in with your username."}
              </p>
            </div>
            {data.password_enabled !== false && <div>
              <label className="text-text-2 text-sm block mb-1.5">New Password <span className="text-text-3">(leave blank to keep current)</span></label>
              <input type="password" name="password" placeholder={`At least ${data.password_min_length ?? 8} characters`}
                minLength={data.password_min_length ?? 8} maxLength={data.password_max_length ?? 128}
                autoComplete="new-password" required={data.password_upgrade_required} aria-describedby="password-requirements"
                className="w-full h-11 px-3 rounded-md bg-inset border border-line text-text focus:outline-none focus:border-accent/50" />
              <p id="password-requirements" className="mt-2 text-xs text-text-2">{data.password_help}</p>
              <label className="text-text-2 text-sm block mt-4 mb-1.5" htmlFor="current-password">Current password <span className="text-text-3">(needed to change your password)</span></label>
              <input id="current-password" type="password" name="current_password" autoComplete="current-password" maxLength={1024}
                className="w-full h-11 px-3 rounded-md bg-inset border border-line text-text focus:outline-none focus:border-accent/50" />
            </div>}
            <Button type="submit" leftIcon={<Check className="w-4 h-4" />}>Save Changes</Button>
          </form>
        </Card>

        {/* Learning preferences */}
        <Card className="mb-5">
          <h2 className="flex items-center gap-2 text-lg"><Sparkles className="w-5 h-5 text-violet" /> Learning Preferences</h2>
          {prefsError && <p role="alert" className="text-danger text-sm mb-3">{prefsError}</p>}
          <p className="text-text-3 text-sm mb-4 -mt-2">Personalize your AI coach recommendations.</p>

          <label className="text-text-2 text-sm block mb-1.5">Preparing for</label>
          <div className="flex flex-wrap gap-2 mb-4">
            {GOALS.map((g) => (
              <button key={g} onClick={() => setGoal(g)}
                className={`px-3 py-1.5 rounded-full text-sm border transition-all ${goal === g ? "border-accent bg-accent/10 text-text" : "border-line text-text-2 hover:border-line-strong"}`}>
                {cap(g)}
              </button>
            ))}
          </div>

          <label className="text-text-2 text-sm block mb-1.5">Learning style</label>
          <div className="flex flex-wrap gap-2 mb-4">
            {STYLES.map((s) => (
              <button key={s} onClick={() => setStyle(s)}
                className={`px-3 py-1.5 rounded-full text-sm border transition-all ${style === s ? "border-accent bg-accent/10 text-text" : "border-line text-text-2 hover:border-line-strong"}`}>
                {cap(s)}
              </button>
            ))}
          </div>

          <label className="text-text-2 text-sm block mb-1.5">Daily study time</label>
          <div className="flex gap-2 mb-5">
            {TIMES.map((m) => (
              <button key={m} onClick={() => setMinutes(m)}
                className={`flex-1 py-2 rounded-md text-sm border transition-all ${minutes === m ? "border-accent bg-accent/10 text-text" : "border-line text-text-2 hover:border-line-strong"}`}>
                {timeLabel(m)}
              </button>
            ))}
          </div>

          <Button onClick={savePrefs} disabled={savingPrefs} leftIcon={prefsSaved ? <Check className="w-4 h-4" /> : undefined}>
            {prefsSaved ? "Saved" : savingPrefs ? "Saving…" : "Save Preferences"}
          </Button>
        </Card>

        {/* Appearance */}
        <Card className="mb-5">
          <h2 className="flex items-center gap-2 text-lg"><Palette className="w-5 h-5 text-accent-2" /> Appearance</h2>
          <div className="flex items-center justify-between mt-3">
            <div>
              <div className="text-sm font-medium">Theme</div>
              <div className="text-text-3 text-sm">Warm paper, clear text, and a little room to focus.</div>
            </div>
            <Badge>Calm workspace</Badge>
          </div>
        </Card>

        <Card className="mb-5">
          <h2 className="text-lg">Account security</h2>
          <p className="text-sm text-text-2 mt-2 mb-4">Manage your contact email, recovery, and linked sign-in providers.</p>
          <a href="/account" className="text-sm font-medium text-accent hover:underline">Account security and recovery</a>
        </Card>

        {/* Data */}
        <Card className="mb-5">
          <h2 className="flex items-center gap-2 text-lg"><Download className="w-5 h-5 text-success" /> Your Data</h2>
          <p className="text-text-3 text-sm mb-4 -mt-2">Download everything we store about your learning.</p>
          <Button variant="secondary" leftIcon={<Download className="w-4 h-4" />} onClick={() => (window.location.href = "/api/export")}>
            Export Data
          </Button>
        </Card>

        {/* Danger zone */}
        <Card className="border-danger/20">
          <h2 className="flex items-center gap-2 text-lg text-danger"><Trash2 className="w-5 h-5" /> Danger Zone</h2>
          <p className="text-text-3 text-sm mb-4 -mt-2">Deleting your account is permanent and removes all your data.</p>
          <form action="/delete_account" method="post" onSubmit={(e) => { if (!confirm("Are you absolutely sure? This is permanent and cannot be undone.")) e.preventDefault(); }}>
            <input type="hidden" name="csrf_token" value={csrfToken()} />
            <button type="submit" className="inline-flex items-center gap-2 h-11 px-5 rounded-md bg-danger/15 text-danger border border-danger/30 hover:bg-danger/25 transition-colors text-sm font-medium">
              <Trash2 className="w-4 h-4" /> Delete My Account
            </button>
          </form>
        </Card>
      </div>
    </AppShell>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-card-2 text-text-2 border border-line">{children}</span>;
}
