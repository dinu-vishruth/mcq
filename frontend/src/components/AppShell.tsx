import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  LayoutDashboard, Compass, FileQuestion, TrendingUp, AlertTriangle,
  RotateCcw, Settings, LogOut, BookOpen, Menu, X,
} from "lucide-react";
import { cn } from "@/lib/cn";

interface NavItem { href: string; label: string; icon: typeof LayoutDashboard; key: string; }

const NAV: NavItem[] = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, key: "dashboard" },
  { href: "/journey", label: "Your library", icon: Compass, key: "journey" },
  { href: "/knowledge", label: "Create a quiz", icon: FileQuestion, key: "knowledge" },
  { href: "/progress", label: "Progress", icon: TrendingUp, key: "progress" },
  { href: "/weak-topics", label: "Topics to revisit", icon: AlertTriangle, key: "weak-topics" },
  { href: "/review", label: "Review mistakes", icon: RotateCcw, key: "review" },
  { href: "/profile", label: "Settings", icon: Settings, key: "settings" },
];

const BRAND = "Examly";

interface Props {
  active: string;
  children: ReactNode;
  username: string;
}

export function AppShell({ active, children, username }: Props) {
  const [open, setOpen] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButton.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        menuButton.current?.focus();
      }
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);
  const initial = (username || "U")[0].toUpperCase();

  return (
    <div className="app-glow min-h-screen">
      {/* Mobile top bar */}
      <div className="lg:hidden sticky top-0 z-30 flex items-center gap-3 px-4 h-14 border-b border-line bg-bg/80 backdrop-blur">
        <button ref={menuButton} onClick={() => setOpen(true)} aria-label="Open menu" aria-expanded={open} aria-controls="study-navigation" className="text-text-2 p-2 -ml-2">
          <Menu className="w-6 h-6" />
        </button>
        <span className="flex items-center gap-2 font-display font-semibold">
          <BookOpen className="w-5 h-5 text-accent" /> {BRAND}
        </span>
      </div>

      {/* Backdrop */}
      {open && <div className="lg:hidden fixed inset-0 z-40 bg-black/50" onClick={() => setOpen(false)} />}

      {/* Sidebar */}
      <aside
        id="study-navigation"
        aria-label="Study navigation"
        className={cn(
          "fixed z-50 top-0 left-0 h-dvh w-[248px] bg-elev border-r border-line flex flex-col p-5 transition-transform duration-200 lg:translate-x-0 lg:visible",
          open ? "translate-x-0 visible" : "-translate-x-full invisible",
        )}
      >
        <div className="flex items-center justify-between pt-2 mb-10">
          <a href="/dashboard" className="flex items-center gap-2.5 font-display font-semibold text-xl tracking-tight">
            <span className="grid place-items-center w-10 h-10 rounded-md bg-accent text-white shadow-glow"><BookOpen className="w-5 h-5" /></span>
            {BRAND}
          </a>
          <button ref={closeButton} onClick={() => { setOpen(false); menuButton.current?.focus(); }} className="lg:hidden text-text-3 p-2" aria-label="Close menu">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="text-text-3 text-xs font-medium uppercase tracking-[0.14em] mb-3 px-3">Your workspace</div>
        <nav aria-label="Main" className="space-y-1.5 overflow-y-auto pb-2">
          {NAV.map((item) => {
            const Icon = item.icon;
            const on = active === item.key;
            return (
              <a
                key={item.key}
                href={item.href}
                aria-current={on ? "page" : undefined}
                onClick={() => setOpen(false)}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors [&_svg]:w-[18px] [&_svg]:h-[18px]",
                  on ? "bg-accent/[0.08] text-accent font-semibold" : "text-text-2 hover:text-text hover:bg-inset",
                )}
              >
                <Icon /> {item.label}
              </a>
            );
          })}
        </nav>

        <div className="flex-1" />

        <div className="my-6 p-4 rounded-lg border border-line bg-card">
          <BookOpen className="w-5 h-5 text-accent mb-2" />
          <p className="font-medium text-sm">One idea at a time.</p>
          <p className="text-text-3 text-xs leading-relaxed mt-1">A little practice makes room for understanding.</p>
        </div>

        <div className="flex items-center gap-3 pt-3 border-t border-line">
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
      <main className="lg:ml-[248px] relative z-10 min-h-screen">
        <div className="hidden lg:flex items-center justify-between gap-4 h-[76px] px-8 xl:px-12 border-b border-line bg-elev/70">
          <p className="text-sm text-text-3">Your workspace <span className="mx-3 text-line-strong">/</span> <span className="text-text font-medium">{NAV.find(item => item.key === active)?.label || "Study space"}</span></p>
          <span className="text-xs text-text-3">Make a little room to learn.</span>
        </div>
        {children}
      </main>
    </div>
  );
}
