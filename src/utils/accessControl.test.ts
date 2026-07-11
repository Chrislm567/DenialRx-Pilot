import { describe, expect, it } from 'vitest';

import { canPerformWorkspaceAction } from './accessControl';

describe('workspace role permissions', () => {
  it('allows admins to manage members', () => {
    expect(canPerformWorkspaceAction('admin', 'manageMembers')).toBe(true);
  });

  it('allows billers to write but not manage members', () => {
    expect(canPerformWorkspaceAction('biller', 'write')).toBe(true);
    expect(canPerformWorkspaceAction('biller', 'manageMembers')).toBe(false);
  });

  it('keeps viewers read only', () => {
    expect(canPerformWorkspaceAction('viewer', 'read')).toBe(true);
    expect(canPerformWorkspaceAction('viewer', 'write')).toBe(false);
  });
});
