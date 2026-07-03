"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { DataTable } from "@/components/data-table";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { useSession } from "@/lib/session";

export default function UploadPage() {
  const router = useRouter();
  const { setFromUpload, profile, preview, filename } = useSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (file: File | null) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.upload(file);
      setFromUpload(result);
      router.push("/overview");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Upload Dataset</h2>
        <p className="mt-2 text-muted-foreground">
          Upload any CSV or Excel file. The system will profile columns, detect types, and prepare the dataset for
          analysis.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Select a file</CardTitle>
          <CardDescription>Supported formats: .csv, .xlsx, .xls (max 50MB)</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            disabled={loading}
            onChange={(e) => handleUpload(e.target.files?.[0] ?? null)}
            className="block w-full text-sm"
          />
          {loading && <p className="text-sm text-muted-foreground">Uploading and profiling...</p>}
          {error && <p className="text-sm text-red-600">{error}</p>}
        </CardContent>
      </Card>

      {profile && preview && (
        <Card>
          <CardHeader>
            <CardTitle>Latest upload: {filename}</CardTitle>
            <CardDescription>
              {profile.shape.rows} rows · {profile.shape.columns} columns · {profile.duplicate_rows} duplicates
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <DataTable rows={preview} />
            <Button onClick={() => router.push("/overview")}>Continue to Overview</Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
