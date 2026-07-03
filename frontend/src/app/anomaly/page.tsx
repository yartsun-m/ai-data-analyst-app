"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { useSession } from "@/lib/session";

export default function AnomalyPage() {
  const router = useRouter();
  const { sessionId } = useSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contamination, setContamination] = useState("0.05");
  const [result, setResult] = useState<{
    anomaly_count: number;
    total_rows: number;
    anomaly_rate: number;
    method: string;
    anomaly_indices: number[];
  } | null>(null);

  useEffect(() => {
    if (!sessionId) router.push("/");
  }, [sessionId, router]);

  const run = async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const response = await api.anomaly(sessionId, Number(contamination));
      setResult(response.anomaly);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Anomaly detection failed");
    } finally {
      setLoading(false);
    }
  };

  if (!sessionId) return <p className="text-muted-foreground">Upload a dataset first.</p>;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Anomaly Detection</h2>
        <p className="mt-2 text-muted-foreground">Isolation Forest flags unusual rows in your dataset.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Run detection</CardTitle>
          <CardDescription>Expected anomaly fraction guides sensitivity.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-end gap-3">
          <div>
            <label className="mb-1 block text-sm text-muted-foreground">Contamination</label>
            <Select value={contamination} onChange={(e) => setContamination(e.target.value)}>
              {["0.01", "0.03", "0.05", "0.1", "0.15", "0.2"].map((v) => (
                <option key={v} value={v}>
                  {Number(v) * 100}%
                </option>
              ))}
            </Select>
          </div>
          <Button onClick={run} disabled={loading}>
            {loading ? "Running..." : "Detect anomalies"}
          </Button>
        </CardContent>
      </Card>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {result && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Anomalies</CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold">{result.anomaly_count}</CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Rate</CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold">
                {(result.anomaly_rate * 100).toFixed(1)}%
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Method</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">{result.method}</CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Sample anomaly row indices</CardTitle>
              <CardDescription>First 100 flagged rows (0-based index in active dataset).</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {result.anomaly_indices.length
                ? result.anomaly_indices.join(", ")
                : "No anomalies detected at this threshold."}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
