import type { Metadata } from "next";

import { NavBar } from "@/components/nav-bar";
import { SessionProvider } from "@/lib/session";

import "./globals.css";

export const metadata: Metadata = {
  title: "AI Data Analyst",
  description: "Upload, profile, clean, analyze, and model any tabular dataset.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <SessionProvider>
          <NavBar />
          <main className="mx-auto max-w-7xl px-4 py-8">{children}</main>
        </SessionProvider>
      </body>
    </html>
  );
}
