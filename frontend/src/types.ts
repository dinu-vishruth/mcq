export interface HistoryItem {
  session_key: string;
  score: number;
  total: number;
  submitted_at: string;
  time_spent: number;
  difficulty: string | null;
}

export interface WeakTopic {
  topic: string;
  wrong: number;
  total: number;
  pct: number;
}

export interface DashboardData {
  username: string;
  total_quizzes: number;
  avg_score: number;
  best_score: number;
  total_time: number;
  streak: number;
  xp: number;
  chart_dates: string[];
  chart_scores: number[];
  difficulty_averages: [number, number, number];
  weak_topics: WeakTopic[];
  history: HistoryItem[];
  recommendations: string[];
  needs_onboarding: boolean;
  prefs: UserPrefs | null;
}

export interface UserPrefs {
  goal: string;
  style: string;
  daily_minutes: number;
}
