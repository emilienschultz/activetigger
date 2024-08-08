import { FC, PropsWithChildren, createContext, useContext, useState } from 'react';

import { NotificationType, ProjectStateModel, SelectionConfig } from '../types';

// Context content
export type AppContextValue = {
  notifications: NotificationType[]; // manage notification
  selectionConfig: SelectionConfig; // selection for the next element
  currentProject?: ProjectStateModel; // current project selected
  reFetchCurrentProject?: () => void; // update the state of the project
  currentScheme?: string; // scheme selected to annotate
};

export const defaultContext: AppContextValue = {
  notifications: [],
  selectionConfig: { mode: 'deterministic', sample: 'untagged' },
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
