"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { DatasetViewer } from "@/components/dataset-viewer";
import { Button } from "@/components/ui/button";
import { useSession } from "@/lib/session";

export default function DatasetPage() {
  const router = useRouter();
  const { sessionId, filename, cleaningReport } = useSession();

  useEffect(() => {
    if (!sessionId) router.push("/");
  }, [sessionId, router]);

  if (!sessionId) {
    return <p className="text-muted-foreground">Upload a dataset first.</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Dataset Viewer</h2>
          <p className="mt-2 text-muted-foreground">
            Spreadsheet-style inspection with server-side pagination — only the current page is loaded.
          </p>
        </div>
        <Button variant="outline" onClick={() => router.push("/overview")}>
          Back to Overview
        </Button>
      </div>

      <DatasetViewer sessionId={sessionId} filename={filename ?? undefined} hasCleaned={!!cleaningReport} />
    </div>
  );
}
