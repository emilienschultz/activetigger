import { createContext, useContext } from 'react';

import { UserModel } from '../types';

export const defaultContext: AppContextValue = {};

export type AppContextValue = {
  user?: UserModel & {
    access_token?: string;
  };
};

export type AppContextType = {
  appContext: AppContextValue;
  setAppContext: React.Dispatch<React.SetStateAction<AppContextValue>>;
};

export const AppContext = createContext(null as unknown as AppContextType);

export function useAppContext() {
  return useContext(AppContext);
}
