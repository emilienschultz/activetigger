import { FC, PropsWithChildren, createContext, useContext, useState } from 'react';

import { NotificationType } from '../types';

export const defaultContext: AppContextValue = { notifications: [] };

export type AppContextValue = {
  notifications: NotificationType[];
};

export type AppContextType = {
  appContext: AppContextValue;
  setAppContext: React.Dispatch<React.SetStateAction<AppContextValue>>;
};

export const AppContext = createContext<AppContextType>(null as unknown as AppContextType);

const _useAppContext = () => {
  const [appContext, setAppContext] = useState<AppContextValue>(defaultContext);
  return {
    appContext,
    setAppContext,
  };
};

export function useAppContext() {
  return useContext(AppContext);
}

export const AppContextProvider: FC<PropsWithChildren> = ({ children }) => {
  const context = _useAppContext();
  return <AppContext.Provider value={context}>{children}</AppContext.Provider>;
};
