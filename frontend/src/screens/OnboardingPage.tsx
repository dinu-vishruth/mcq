import { Onboarding } from "./Onboarding";
import type { UserPrefs } from "@/types";

export default function OnboardingPage({ data }: { data: { prefs: UserPrefs | null } }) {
  return <Onboarding initial={data.prefs} onDone={() => window.location.assign("/dashboard")} />;
}
