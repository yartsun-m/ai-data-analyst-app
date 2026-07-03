"use client";

import { DatasetViewer } from "@/components/dataset-viewer";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface DatasetViewerModalProps {
  open: boolean;
  onClose: () => void;
  sessionId: string;
  filename?: string;
  hasCleaned?: boolean;
}

export function DatasetViewerModal({
  open,
  onClose,
  sessionId,
  filename,
  hasCleaned,
}: DatasetViewerModalProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <button
        type="button"
        className="absolute inset-0 bg-black/50"
        aria-label="Close dataset viewer"
        onClick={onClose}
      />
      <div
        className={cn(
          "relative z-10 flex max-h-[92vh] w-full max-w-[96vw] flex-col overflow-hidden rounded-lg border bg-background shadow-xl",
        )}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-viewer-title"
      >
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div>
            <h2 id="dataset-viewer-title" className="text-lg font-semibold">
              Full dataset
            </h2>
            <p className="text-sm text-muted-foreground">Browse, search, sort, and paginate all rows and columns.</p>
          </div>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </div>
        <div className="overflow-y-auto px-4 py-4">
          <DatasetViewer sessionId={sessionId} filename={filename} hasCleaned={hasCleaned} />
        </div>
      </div>
    </div>
  );
}
