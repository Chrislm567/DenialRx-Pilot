'use client';

type ExportButtonProps = {
  filename: string;
  data: Record<string, unknown>[];
};

export function ExportButton({ filename, data }: ExportButtonProps) {
  const handleClick = () => {
    if (!data.length) return;
    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];
    data.forEach((row) => {
      const values = headers.map((header) => {
        const cell = row[header];
        if (cell === null || cell === undefined) return '';
        if (typeof cell === 'number') return cell.toString();
        const safe = String(cell).replace(/"/g, '""');
        return `"${safe}"`;
      });
      csvRows.push(values.join(','));
    });

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    link.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <button type="button" className="csv-button" onClick={handleClick}>
      Export CSV
    </button>
  );
}
