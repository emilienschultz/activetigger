import { createContext, useContext } from 'react';

export type AppContextType = {
  // TODO
};

export const AppContext = createContext(null as unknown as AppContextType);

export function useAppContext() {
  return useContext(AppContext);
}
