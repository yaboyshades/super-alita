export function isEnabled(key: string): boolean {
  const v = (process.env[key] ?? '').toString().trim().toLowerCase();
  return v === '1' || v === 'true' || v === 'yes' || v === 'on';
}
