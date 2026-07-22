'use client';

import { createContext, useContext } from 'react';

// FE-001 ROOT FIX: added optional `name` field so DiseaseSearchScreen can
// pass the disease name to SearchResultsScreen, which then queries the real
// RL ranker by disease name.
export type Route = { page: string; section?: string; sub?: string; id?: string; name?: string };

type NavContext = {
  navigate: (route: Route) => void;
  currentRoute: Route;
};

export const DrugOSNavContext = createContext<NavContext>({
  navigate: () => {},
  currentRoute: { page: 'app', section: 'search' },
});

export function useDrugOSNav() {
  return useContext(DrugOSNavContext);
}
