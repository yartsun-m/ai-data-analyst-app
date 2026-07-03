"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { PlotlyChart } from "@/components/plotly-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { useSession } from "@/lib/session";

export default function ClusteringPage() {
  const router = useRouter();
  const { sessionId } = useSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nClusters, setNClusters] = useState<string>("auto");
  const [result, setResult] = useState<{
    n_clusters: number;
    silhouette_score: number | null;
    cluster_sizes: Record<string, number>;
    scatter: Array<{ x: number; y: number; cluster: number }>;
  } | null>(null);

  useEffect(() => {
    if (!sessionId) router.push("/");
  }, [sessionId, router]);

  const scatterFigure = useMemo(() => {
    if (!result?.scatter?.length) return null;
    const clusters = [...new Set(result.scatter.map((p) => p.cluster))];
    const traces = clusters.map((cluster) => {
      const points = result.scatter.filter((p) => p.cluster === cluster);
      return {
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        mode: "markers" as const,
        type: "scatter" as const,
        name: `Cluster ${cluster}`,
      };
    });
    return { data: traces, layout: { title: "K-Means clusters (PCA 2D)", height: 420 } };
  }, [result]);

  const run = async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const response = await api.clustering(
        sessionId,
        nClusters === "auto" ? undefined : Number(nClusters),
      );
      setResult(response.clustering);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Clustering failed");
    } finally {
      setLoading(false);
    }
  };

  if (!sessionId) return <p className="text-muted-foreground">Upload a dataset first.</p>;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Clustering</h2>
        <p className="mt-2 text-muted-foreground">K-Means segmentation with silhouette-based k selection.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Run clustering</CardTitle>
          <CardDescription>Uses numeric and encoded features; visualized with PCA.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-end gap-3">
          <div>
            <label className="mb-1 block text-sm text-muted-foreground">Clusters</label>
            <Select value={nClusters} onChange={(e) => setNClusters(e.target.value)}>
              <option value="auto">Auto (silhouette)</option>
              {[2, 3, 4, 5, 6, 7, 8].map((k) => (
                <option key={k} value={String(k)}>
                  {k}
                </option>
              ))}
            </Select>
          </div>
          <Button onClick={run} disabled={loading}>
            {loading ? "Running..." : "Run K-Means"}
          </Button>
        </CardContent>
      </Card>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {result && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Clusters</CardTitle>
            </CardHeader>
            <CardContent className="text-2xl font-semibold">{result.n_clusters}</CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Silhouette</CardTitle>
            </CardHeader>
            <CardContent className="text-2xl font-semibold">
              {result.silhouette_score?.toFixed(3) ?? "N/A"}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Sizes</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {Object.entries(result.cluster_sizes)
                .map(([k, v]) => `C${k}: ${v}`)
                .join(" · ")}
            </CardContent>
          </Card>
        </div>
      )}

      {scatterFigure && (
        <Card>
          <CardHeader>
            <CardTitle>Cluster scatter</CardTitle>
          </CardHeader>
          <CardContent>
            <PlotlyChart figure={scatterFigure} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
