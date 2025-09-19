'use client';

import Link from 'next/link';
import { ReactNode } from 'react';
import { usePathname } from 'next/navigation';

const navItems = [
  { href: '/overview', label: 'Overview' },
  { href: '/claims-explorer', label: 'Claims Explorer' },
  { href: '/payer-scorecards', label: 'Payer Scorecards' }
];

function Navigation() {
  const pathname = usePathname();
  return (
    <nav className="nav-links">
      {navItems.map((item) => {
        const isActive = pathname === item.href;
        return (
          <Link key={item.href} href={item.href} className={isActive ? 'active' : ''}>
            {item.label}
          </Link>
        );
      })}
      <Link href="/api/auth/logout">Sign out</Link>
    </nav>
  );
}

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>Phase Runner</h1>
        <p>Denial analytics & observability</p>
        <Navigation />
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}
