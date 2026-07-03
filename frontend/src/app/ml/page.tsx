"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { useSession } from "@/lib/session";

export default function MLPage() {
  const router = useRouter();
  const { sessionId, profile, targetColumn, setTargetColumn, mlResults, setMLResults } = useSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [targetOptions, setTargetOptions] = useState<string[]>([]);

  useEffect(() => {
    if (!sessionId) router.push("/");
  }, [sessionId, router]);

  useEffect(() => {
    if (!sessionId || !profile) return;
    api.profile(sessionId).then((result) => {
      const active = result.active_columns ?? profile.columns;
      const suggested = profile.suggested_targets.filter((col) => active.includes(col));
      setTargetOptions(suggested.length ? suggested : active);
    }).catch(() => {
      setTargetOptions(profile.columns);
    });
  }, [sessionId, profile]);

  if (!sessionId || !profile) {
    return <p className="text-muted-foreground">Upload a dataset first.</p>;
  }

  const handleTrain = async () => {
    if (!targetColumn) {
      setError("Select a target column before training.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const started = await api.train(sessionId, targetColumn, true);
      if (started.ml_results) {
        setMLResults(started.ml_results);
        return;
      }
      if (!started.job_id) throw new Error("No job id returned");
      let attempts = 0;
      while (attempts < 120) {
        await new Promise((r) => setTimeout(r, 1500));
        const status = await api.trainStatus(started.job_id);
        if (status.status === "completed" && status.ml_results) {
          setMLResults(status.ml_results);
          return;
        }
        if (status.status === "failed") {
          throw new Error(status.error || "Training failed");
        }
        attempts += 1;
      }
      throw new Error("Training timed out");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Machine Learning</h2>
        <p className="mt-2 text-muted-foreground">AutoML lite: train and compare models on your selected target.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Train Models</CardTitle>
          <CardDescription>
            Suggested targets: {profile.suggested_targets.join(", ") || "none"}
            {profile.ml_excluded_columns && profile.ml_excluded_columns.length > 0 && (
              <span className="mt-1 block text-xs">
                Excluded from features: IDs, phones, emails, names ({profile.ml_excluded_columns.length} columns)
              </span>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-3">
          <Select
            value={targetColumn || ""}
            onChange={(e) => setTargetColumn(e.target.value || null)}
            className="max-w-xs"
          >
            <option value="">Select target</option>
            {targetOptions.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </Select>
          <Button onClick={handleTrain} disabled={loading}>
            {loading ? "Training..." : "Run AutoML"}
          </Button>
        </CardContent>
      </Card>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {mlResults?.warnings && mlResults.warnings.length > 0 && (
        <Card className="border-amber-300 bg-amber-50/50">
          <CardHeader>
            <CardTitle className="text-base">Notes</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc space-y-1 pl-5 text-sm text-amber-950">
              {mlResults.warnings.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {mlResults && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardDescription>Task</CardDescription>
                <CardTitle>{mlResults.task_type}</CardTitle>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardDescription>Best Model</CardDescription>
                <CardTitle>{mlResults.best_model}</CardTitle>
              </CardHeader>
            </Card>
            <Card
              className={
                mlResults.task_type === "regression" &&
                mlResults.best_metrics.r2 != null &&
                mlResults.best_metrics.r2 < 0
                  ? "border-amber-300 bg-amber-50/50"
                  : undefined
              }
            >
              <CardHeader>
                <CardDescription>
                  Primary Metric
                  {mlResults.task_type === "regression" &&
                    mlResults.best_metrics.r2 != null &&
                    mlResults.best_metrics.r2 < 0 &&
                    " (below baseline)"}
                </CardDescription>
                <CardTitle>
                  {mlResults.best_metrics.r2 ??
                    mlResults.best_metrics.f1_weighted ??
                    mlResults.best_metrics.accuracy ??
                    "—"}
                </CardTitle>
              </CardHeader>
            </Card>
          </div>

          {mlResults.cross_validation?.mean != null && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Cross-Validation</CardTitle>
                <CardDescription>
                  {mlResults.cross_validation.scoring}: {mlResults.cross_validation.mean.toFixed(4)} ±{" "}
                  {(mlResults.cross_validation.std ?? 0).toFixed(4)}
                </CardDescription>
              </CardHeader>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Model Leaderboard</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {mlResults.leaderboard.map((entry) => (
                <div key={entry.model} className="rounded-md border p-3 text-sm">
                  <p className="font-medium">{entry.model}</p>
                  {entry.metrics && (
                    <p className="text-muted-foreground">{JSON.stringify(entry.metrics)}</p>
                  )}
                  {entry.error && <p className="text-red-600">{entry.error}</p>}
                </div>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Feature Importance</CardTitle>
              <CardDescription>
                Method: {mlResults.explainability?.method || "feature_importance"} · aggregated by source column
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {(mlResults.explainability?.top_features || mlResults.feature_importance).map((item) => (
                  <div key={item.feature} className="flex items-center justify-between rounded border px-3 py-2 text-sm">
                    <span>{item.feature}</span>
                    <span>
                      {item.importance != null
                        ? `${(item.importance * 100).toFixed(1)}%`
                        : item.mean_abs_shap != null
                          ? `${(item.mean_abs_shap * 100).toFixed(1)}%`
                          : "—"}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}

      <Button variant="secondary" onClick={() => router.push("/chat")}>
        Ask the AI Assistant
      </Button>
    </div>
  );
}
