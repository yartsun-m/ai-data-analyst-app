"use client";

import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

import type { CleaningReport, MLResults, ProfileData, UploadResponse } from "@/lib/api";

interface SessionState {
  sessionId: string | null;
  filename: string | null;
  profile: ProfileData | null;
  preview: Record<string, unknown>[] | null;
  cleaningReport: CleaningReport | null;
  mlResults: MLResults | null;
  targetColumn: string | null;
  setFromUpload: (data: UploadResponse) => void;
  setProfile: (profile: ProfileData, preview?: Record<string, unknown>[]) => void;
  setCleaningReport: (report: CleaningReport) => void;
  setMLResults: (results: MLResults) => void;
  setTargetColumn: (column: string | null) => void;
}

const SessionContext = createContext<SessionState | null>(null);

const STORAGE_KEY = "ai-data-analyst-session";

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [profile, setProfileState] = useState<ProfileData | null>(null);
  const [preview, setPreview] = useState<Record<string, unknown>[] | null>(null);
  const [cleaningReport, setCleaningReportState] = useState<CleaningReport | null>(null);
  const [mlResults, setMLResultsState] = useState<MLResults | null>(null);
  const [targetColumn, setTargetColumnState] = useState<string | null>(null);

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw);
      setSessionId(parsed.sessionId ?? null);
      setFilename(parsed.filename ?? null);
      setProfileState(parsed.profile ?? null);
      setPreview(parsed.preview ?? null);
      setCleaningReportState(parsed.cleaningReport ?? null);
      setMLResultsState(parsed.mlResults ?? null);
      setTargetColumnState(parsed.targetColumn ?? null);
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ sessionId, filename, profile, preview, cleaningReport, mlResults, targetColumn }),
    );
  }, [sessionId, filename, profile, preview, cleaningReport, mlResults, targetColumn]);

  const value = useMemo<SessionState>(
    () => ({
      sessionId,
      filename,
      profile,
      preview,
      cleaningReport,
      mlResults,
      targetColumn,
      setFromUpload: (data) => {
        setSessionId(data.session_id);
        setFilename(data.filename);
        setProfileState(data.profile);
        setPreview(data.preview);
        setCleaningReportState(null);
        setMLResultsState(null);
        setTargetColumnState(data.profile.target_column);
      },
      setProfile: (nextProfile, nextPreview) => {
        setProfileState(nextProfile);
        if (nextPreview) setPreview(nextPreview);
      },
      setCleaningReport: setCleaningReportState,
      setMLResults: setMLResultsState,
      setTargetColumn: setTargetColumnState,
    }),
    [sessionId, filename, profile, preview, cleaningReport, mlResults, targetColumn],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used within SessionProvider");
  return ctx;
}
