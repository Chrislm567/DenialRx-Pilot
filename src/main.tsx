import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { AuthGate } from './components/AuthGate';
import { Dashboard } from './components/Dashboard';
import './index.css';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('DenialRx root element was not found.');
}

createRoot(rootElement).render(
  <StrictMode>
    <AuthGate>
      <Dashboard />
    </AuthGate>
  </StrictMode>,
);
