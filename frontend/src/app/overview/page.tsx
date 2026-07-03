"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { DataTable } from "@/components/data-table";
import { DatasetViewerModal } from "@/components/dataset-viewer-modal";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { useSession } from "@/lib/session";

export default function OverviewPage() {
  const router = useRouter();
  const {
    sessionId,
    profile,
    preview,
    filename,
    cleaningReport,
    targetColumn,
    setProfile,
    setCleaningReport,
    setTargetColumn,
  } = useSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewerOpen, setViewerOpen] = useState(false);
  const [outlierStrategy, setOutlierStrategy] = useState("winsorize");

  useEffect(() => {
    if (!sessionId) router.push("/");
  }, [sessionId, router]);

  if (!sessionId || !profile) {
    return <p className="text-muted-foreground">Upload a dataset first.</p>;
  }

  const handleProfileRefresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.profile(sessionId, targetColumn || undefined);
      setProfile(result.profile, result.preview);
      setTargetColumn(result.profile.target_column);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh profile");
    } finally {
      setLoading(false);
    }
  };

  const handleClean = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.clean(sessionId, targetColumn || undefined, outlierStrategy);
      setCleaningReport(result.cleaning_report);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Cleaning failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Dataset Overview</h2>
        <p className="mt-2 text-muted-foreground">Profiling summary, type detection, and automatic cleaning.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        {[
          ["Rows", profile.shape.rows],
          ["Columns", profile.shape.columns],
          ["Missing Cells", profile.total_missing_cells],
          ["Duplicates", profile.duplicate_rows],
        ].map(([label, value]) => (
          <Card key={label as string}>
            <CardHeader className="pb-2">
              <CardDescription>{label as string}</CardDescription>
              <CardTitle className="text-2xl">{value as number}</CardTitle>
            </CardHeader>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Target Column (optional)</CardTitle>
          <CardDescription>Select a target to enable ML task detection.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-3">
          <Select
            value={targetColumn || ""}
            onChange={(e) => setTargetColumn(e.target.value || null)}
            className="max-w-xs"
          >
            <option value="">None</option>
            {profile.columns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </Select>
          <Button variant="outline" onClick={handleProfileRefresh} disabled={loading}>
            Update Profile
          </Button>
          <Button variant="outline" onClick={() => api.exportDataset(sessionId, cleaningReport ? "cleaned" : "raw", filename ?? undefined)}>
            Export CSV
          </Button>
          <div>
            <label className="mb-1 block text-xs text-muted-foreground">Outlier treatment</label>
            <Select value={outlierStrategy} onChange={(e) => setOutlierStrategy(e.target.value)}>
              <option value="none">None</option>
              <option value="winsorize">Winsorize (1–99%)</option>
              <option value="clip">Clip (IQR)</option>
              <option value="remove">Remove (IQR)</option>
            </Select>
          </div>
          <Button onClick={handleClean} disabled={loading}>
            Run Cleaning Pipeline
          </Button>
        </CardContent>
      </Card>

      {profile.validation_report && (
        <Card className={profile.validation_report.passed ? "" : "border-amber-300 bg-amber-50/50"}>
          <CardHeader>
            <CardTitle>Data Validation</CardTitle>
            <CardDescription>
              {profile.validation_report.passed ? "Checks passed" : "Issues detected"} ·{" "}
              {profile.validation_report.checks_run} checks on {profile.validation_report.rows_validated} rows
            </CardDescription>
          </CardHeader>
          {profile.validation_report.issues?.length > 0 && (
            <CardContent>
              <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
                {profile.validation_report.issues.map((issue: string) => (
                  <li key={issue}>{issue}</li>
                ))}
              </ul>
            </CardContent>
          )}
          {profile.validation_report.quality_report?.summary && (
            <CardContent className="border-t pt-4 text-sm text-muted-foreground">
              Quality summary: {profile.validation_report.quality_report.summary.total_columns} columns ·{" "}
              {profile.validation_report.quality_report.summary.columns_with_high_missing} high-missing ·{" "}
              {profile.validation_report.quality_report.summary.identifier_like_columns} identifier-like
            </CardContent>
          )}
        </Card>
      )}

      {profile.task_type && (
        <Card>
          <CardHeader>
            <CardTitle>Detected ML Task</CardTitle>
            <CardDescription>{profile.task_type}</CardDescription>
          </CardHeader>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Column Types</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 md:grid-cols-2">
            {Object.entries(profile.column_types).map(([col, type]) => (
              <div key={col} className="flex items-center justify-between rounded-md border px-3 py-2 text-sm">
                <span>{col}</span>
                <span className="rounded bg-muted px-2 py-1 text-xs uppercase">
                  {profile.column_roles?.[col] ? `${profile.column_roles[col]} / ` : ""}
                  {type}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {cleaningReport && (
        <Card>
          <CardHeader>
            <CardTitle>Cleaning Results</CardTitle>
            <CardDescription>
              {cleaningReport.rows_before} → {cleaningReport.rows_after} rows · {cleaningReport.columns_before} →{" "}
              {cleaningReport.columns_after} columns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
              {cleaningReport.steps.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {preview && (
        <Card>
          <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle>Preview</CardTitle>
              <CardDescription>
                Lightweight sample ({preview.length} rows). Open the full viewer to browse every row and column.
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setViewerOpen(true)}>
                Open full dataset
              </Button>
              <Button variant="secondary" onClick={() => router.push("/dataset")}>
                Full page viewer
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <DataTable rows={preview} />
          </CardContent>
        </Card>
      )}

      <DatasetViewerModal
        open={viewerOpen}
        onClose={() => setViewerOpen(false)}
        sessionId={sessionId}
        filename={filename ?? undefined}
        hasCleaned={!!cleaningReport}
      />

      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="flex gap-3">
        <Button variant="secondary" onClick={() => router.push("/eda")}>
          Go to EDA
        </Button>
        <Button onClick={() => router.push("/ml")}>Go to ML</Button>
      </div>
    </div>
  );
}
