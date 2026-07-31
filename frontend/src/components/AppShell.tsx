import { useState, type ReactNode } from "react";
import {
  LayoutDashboard, Compass, FileQuestion, TrendingUp, AlertTriangle,
  Trophy, Settings, LogOut, GraduationCap, Flame, Zap, Menu, X,
} from "lucide-react";
import { cn } from "@/lib/cn";

interface NavItem { href: string; label: string; icon: typeof LayoutDashboard; key: string; }

const NAV: NavItem[] = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, key: "dashboard" },
  { href: "/journey", label: "Learning Journey", icon: Compass, key: "journey" },
  { href: "/knowledge", label: "Make Quiz & Test", icon: FileQuestion, key: "knowledge" },
  { href: "/progress", label: "Progress", icon: TrendingUp, key: "progress" },
  { href: "/weak-topics", label: "Weak Topics", icon: AlertTriangle, key: "weak-topics" },
  { href: "/achievements", label: "Achievements", icon: Trophy, key: "achievements" },
  { href: "/profile", label: "Settings", icon: Settings, key: "settings" },
];

const BRAND = "Examly";

interface Props {
  active: string;
  children: ReactNode;
  username: string;
  streak?: number;
  xp?: number;
}

export function AppShell({ active, children, username, streak = 0, xp = 0 }: Props) {
  const [open, setOpen] = useState(false);
  const initial = (username || "U")[0].toUpperCase();

  return (
    <div className="app-glow min-h-screen">
      {/* Mobile top bar */}
      <div className="lg:hidden sticky top-0 z-30 flex items-center gap-3 px-4 h-14 border-b border-white/[0.06] bg-bg/80 backdrop-blur">
        <button onClick={() => setOpen(true)} aria-label="Open menu" className="text-text-2">
          <Menu className="w-6 h-6" />
        </button>
        <span className="flex items-center gap-2 font-display font-semibold">
          <GraduationCap className="w-5 h-5 text-accent" /> {BRAND}
        </span>
      </div>

      {/* Backdrop */}
      {open && <div className="lg:hidden fixed inset-0 z-40 bg-black/50" onClick={() => setOpen(false)} />}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed z-50 top-0 left-0 h-full w-[264px] bg-elev border-r border-white/[0.06] flex flex-col p-4 transition-transform lg:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center justify-between mb-6">
          <a href="/dashboard" className="flex items-center gap-2.5 font-display font-semibold text-[1.05rem]">
            <span className="grid place-items-center w-9 h-9 rounded-md bg-accent/10 text-accent"><GraduationCap className="w-5 h-5" /></span>
            {BRAND}
          </a>
          <button onClick={() => setOpen(false)} className="lg:hidden text-text-3" aria-label="Close menu">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="text-text-3 text-xs uppercase tracking-wider mb-2 px-2">Menu</div>
        <nav className="space-y-1">
          {NAV.map((item) => {
            const Icon = item.icon;
            const on = active === item.key;
            return (
              <a
                key={item.key}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors [&_svg]:w-[18px] [&_svg]:h-[18px]",
                  on ? "bg-accent/12 text-text font-medium" : "text-text-2 hover:text-text hover:bg-white/[0.04]",
                )}
              >
                <Icon /> {item.label}
              </a>
            );
          })}
        </nav>

        <div className="flex-1" />

        {/* Streak + XP mini-stats */}
        <div className="space-y-2 mb-3">
          <div className="flex items-center gap-3 px-3 py-2.5 rounded-md bg-white/[0.03]">
            <span className="grid place-items-center w-8 h-8 rounded-sm bg-warning/10 text-warning"><Flame className="w-4 h-4" /></span>
            <div>
              <div className="text-sm font-medium">{streak} day streak</div>
              <div className="text-text-3 text-xs">Keep it going</div>
            </div>
          </div>
          <div className="flex items-center gap-3 px-3 py-2.5 rounded-md bg-white/[0.03]">
            <span className="grid place-items-center w-8 h-8 rounded-sm bg-violet/10 text-violet"><Zap className="w-4 h-4" /></span>
            <div>
              <div className="text-sm font-medium">{xp} XP</div>
              <div className="text-text-3 text-xs">Level {Math.floor(xp / 100) + 1}</div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 pt-3 border-t border-white/[0.06]">
          <a href="/profile" className="flex items-center gap-3 flex-1 min-w-0">
            <span className="grid place-items-center w-9 h-9 rounded-md bg-accent/15 text-accent font-semibold shrink-0">{initial}</span>
            <div className="min-w-0">
              <div className="text-sm font-medium truncate">{username}</div>
              <div className="text-text-3 text-xs">View profile</div>
            </div>
          </a>
          <a href="/logout" aria-label="Log out" className="text-text-3 hover:text-danger transition-colors p-2">
            <LogOut className="w-[18px] h-[18px]" />
          </a>
        </div>
      </aside>

      {/* Main content */}
      <main className="lg:ml-[264px] relative z-10">{children}</main>
    </div>
  );
}
