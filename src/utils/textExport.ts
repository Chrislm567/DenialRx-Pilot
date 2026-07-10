const sanitizeFileSegment = (value: string): string => {
  const sanitized = value.trim().replace(/[^a-zA-Z0-9_-]+/g, '_');
  return sanitized || 'MISSING_ID';
};

export function exportAppealText(claimId: string, letterText: string): void {
  const documentText = letterText.trim() || '[MISSING APPEAL CONTENT]';
  const blob = new Blob([documentText], { type: 'text/plain;charset=utf-8' });
  const downloadUrl = URL.createObjectURL(blob);
  const downloadLink = document.createElement('a');

  downloadLink.href = downloadUrl;
  downloadLink.download = `Appeal_Claim_${sanitizeFileSegment(claimId)}.txt`;
  downloadLink.style.display = 'none';

  document.body.appendChild(downloadLink);
  downloadLink.click();
  downloadLink.remove();
  URL.revokeObjectURL(downloadUrl);
}
