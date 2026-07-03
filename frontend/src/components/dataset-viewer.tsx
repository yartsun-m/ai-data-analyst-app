"use client";

import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { api, DatasetColumn, DatasetPageResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

const PAGE_SIZE_OPTIONS = [25, 50, 100, 200] as const;

type DatasetVariant = "raw" | "cleaned";

interface DatasetViewerProps {
  sessionId: string;
  filename?: string;
  hasCleaned?: boolean;
  defaultVariant?: DatasetVariant;
  className?: string;
}

function formatCell(value: unknown, type: string): string {
  if (value === null || value === undefined || value === "") return "—";
  if (type === "boolean") return value ? "true" : "false";
  if (type === "numeric" && typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toLocaleString(undefined, { maximumFractionDigits: 6 });
  }
  return String(value);
}

function columnAlign(type: string): string {
  return type === "numeric" ? "text-right tabular-nums" : "text-left";
}

export function DatasetViewer({
  sessionId,
  filename,
  hasCleaned = false,
  defaultVariant = "raw",
  className,
}: DatasetViewerProps) {
  const [variant, setVariant] = useState<DatasetVariant>(defaultVariant);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState<number>(50);
  const [sortBy, setSortBy] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [data, setData] = useState<DatasetPageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cleanedAvailable, setCleanedAvailable] = useState(hasCleaned);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSearch(searchInput.trim());
      setPage(1);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

  const loadPage = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.dataset(sessionId, {
        page,
        pageSize,
        sortBy: sortBy ?? undefined,
        sortOrder,
        search: search || undefined,
        variant,
      });
      setData(result);
      if (result.has_cleaned) setCleanedAvailable(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dataset");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [sessionId, page, pageSize, sortBy, sortOrder, search, variant]);

  useEffect(() => {
    loadPage();
  }, [loadPage]);

  const columns: DatasetColumn[] = data?.columns ?? [];

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(column);
      setSortOrder("asc");
    }
    setPage(1);
  };

  const rowStart = data ? data.row_offset + 1 : 0;
  const rowEnd = data ? Math.min(data.row_offset + data.rows.length, data.filtered_rows) : 0;

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <div className="flex flex-wrap items-end gap-3">
        <div className="min-w-[220px] flex-1">
          <label className="mb-1 block text-xs font-medium text-muted-foreground">Search all columns</label>
          <Input
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Filter rows…"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-muted-foreground">Rows per page</label>
          <Select
            value={String(pageSize)}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
            className="min-w-[100px]"
          >
            {PAGE_SIZE_OPTIONS.map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </Select>
        </div>
        {cleanedAvailable && (
          <div>
            <label className="mb-1 block text-xs font-medium text-muted-foreground">Dataset</label>
            <Select
              value={variant}
              onChange={(e) => {
                setVariant(e.target.value as DatasetVariant);
                setPage(1);
              }}
              className="min-w-[140px]"
            >
              <option value="raw">Original</option>
              <option value="cleaned">Cleaned</option>
            </Select>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-muted-foreground">
        <p>
          {filename ? <span className="font-medium text-foreground">{filename}</span> : "Dataset"}
          {data && (
            <>
              {" · "}
              Showing {rowStart.toLocaleString()}–{rowEnd.toLocaleString()} of{" "}
              {data.filtered_rows.toLocaleString()}
              {data.search ? ` matching “${data.search}”` : ""}
              {data.filtered_rows !== data.total_rows && (
                <> (from {data.total_rows.toLocaleString()} total rows)</>
              )}
              {" · "}
              {columns.length} columns
            </>
          )}
        </p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={loading || !data || data.page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            Previous
          </Button>
          <span className="min-w-[120px] text-center">
            Page {data?.page ?? 1} of {data?.total_pages ?? 1}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={loading || !data || data.page >= data.total_pages}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </Button>
        </div>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="relative overflow-hidden rounded-md border bg-white shadow-sm">
        <div className="max-h-[65vh] overflow-auto">
          <table className="min-w-max border-collapse text-sm">
            <thead className="sticky top-0 z-10 bg-muted/95 backdrop-blur">
              <tr>
                <th className="sticky left-0 z-20 min-w-[56px] border-b border-r bg-muted/95 px-3 py-2 text-left font-medium text-muted-foreground">
                  #
                </th>
                {columns.map((col) => {
                  const active = sortBy === col.name;
                  return (
                    <th
                      key={col.name}
                      className="cursor-pointer select-none border-b px-3 py-2 text-left font-medium whitespace-nowrap hover:bg-muted"
                      onClick={() => handleSort(col.name)}
                      title={`Sort by ${col.name}`}
                    >
                      <span className="inline-flex items-center gap-1">
                        {col.name}
                        <span className="text-[10px] uppercase text-muted-foreground">{col.type}</span>
                        {active && <span aria-hidden>{sortOrder === "asc" ? "↑" : "↓"}</span>}
                      </span>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {loading && !data?.rows.length ? (
                <tr>
                  <td colSpan={Math.max(columns.length + 1, 2)} className="px-3 py-8 text-center text-muted-foreground">
                    Loading…
                  </td>
                </tr>
              ) : data?.rows.length ? (
                data.rows.map((row, idx) => (
                  <tr key={`${data.row_offset}-${idx}`} className="border-t hover:bg-muted/30">
                    <td className="sticky left-0 z-[1] border-r bg-white px-3 py-2 text-muted-foreground tabular-nums">
                      {(data.row_offset + idx + 1).toLocaleString()}
                    </td>
                    {columns.map((col) => (
                      <td
                        key={col.name}
                        className={cn(
                          "max-w-[320px] truncate px-3 py-2 whitespace-nowrap",
                          columnAlign(col.type),
                        )}
                        title={formatCell(row[col.name], col.type)}
                      >
                        {formatCell(row[col.name], col.type)}
                      </td>
                    ))}
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={Math.max(columns.length + 1, 2)} className="px-3 py-8 text-center text-muted-foreground">
                    No rows match your filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {loading && !!data?.rows.length && (
          <div className="pointer-events-none absolute inset-0 bg-white/40" aria-hidden />
        )}
      </div>
    </div>
  );
}
