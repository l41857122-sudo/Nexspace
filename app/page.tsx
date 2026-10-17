import LandingPageContainer from "./components/landing/LandingPageContainer";

export const metadata = {
  title: "NexSpace — Autonomous Remote Sensing Intelligence",
  description:
    "Cinematic orbital remote sensing platform. Transform satellite pixels into verified spatial intelligence through natural language.",
};

export default function Home() {
  return (
    <main className="min-h-screen bg-[#010308]">
      <LandingPageContainer />
    </main>
  );
}