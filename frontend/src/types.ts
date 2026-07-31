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

export interface KnowledgeItem {
  id: number;
  title: string;
  subject: string;
  created_at: string;
  indexed: boolean;
  status: string;
  topic_count: number;
  est_minutes: number;
}

export interface Recommendation {
  kind: string;
  title: string;
  reason: string;
  cta: string;
}

export interface RecentSession {
  session_key: string;
  score: number;
  total: number;
  submitted_at: string;
  difficulty: string | null;
  pct: number;
}

export interface DashboardApi {
  username: string;
  total_quizzes: number;
  avg_score: number;
  total_time: number;
  streak: number;
  xp: number;
  level: number;
  knowledge_count: number;
  goal: string;
  daily_minutes: number;
  chart: { date: string; score: number }[];
  weak_topics: WeakTopic[];
  recent: RecentSession[];
  recommendations: Recommendation[];
  needs_onboarding: boolean;
}
