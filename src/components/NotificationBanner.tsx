export type NotificationTone = 'success' | 'warning';

interface NotificationBannerProps {
  message: string;
  tone: NotificationTone;
  onDismiss: () => void;
}

const toneClasses: Record<NotificationTone, string> = {
  success: 'border-emerald-200 bg-emerald-50 text-emerald-900',
  warning: 'border-amber-200 bg-amber-50 text-amber-900',
};

export function NotificationBanner({
  message,
  tone,
  onDismiss,
}: NotificationBannerProps) {
  return (
    <div
      role="status"
      className={`flex items-center justify-between gap-4 rounded-lg border px-4 py-3 text-sm ${toneClasses[tone]}`}
    >
      <p className="font-medium">{message}</p>
      <button type="button" onClick={onDismiss} className="font-semibold underline-offset-2 hover:underline">
        Dismiss
      </button>
    </div>
  );
}
