"use client";

interface DataTableProps {
  rows: Record<string, unknown>[];
  maxCols?: number;
}

export function DataTable({ rows, maxCols = 8 }: DataTableProps) {
  if (!rows.length) return <p className="text-sm text-muted-foreground">No preview available.</p>;

  const columns = Object.keys(rows[0]).slice(0, maxCols);

  return (
    <div className="overflow-auto rounded-md border">
      <table className="min-w-full text-left text-sm">
        <thead className="bg-muted/50">
          <tr>
            {columns.map((col) => (
              <th key={col} className="px-3 py-2 font-medium">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 15).map((row, idx) => (
            <tr key={idx} className="border-t">
              {columns.map((col) => (
                <td key={col} className="px-3 py-2 text-muted-foreground">
                  {String(row[col] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
