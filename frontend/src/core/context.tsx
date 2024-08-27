import { FC, PropsWithChildren, createContext, useContext, useEffect, useState } from 'react';

import { NotificationType, ProjectStateModel, SelectionConfig } from '../types';

// Context content
export type AppContextValue = {
  notifications: NotificationType[]; // manage notification
  selectionConfig: SelectionConfig; // selection for the next element
  currentProject?: ProjectStateModel | null; // current project selected
  currentScheme?: string; // scheme selected to annotate
  currentProjection?: string;
  freqRefreshSimpleModel: number; // freq to refresh active learning model
  history: string[]; // element annotated
  reFetchCurrentProject?: () => void; // update the state of the project
};

const CONTEXT_LOCAL_STORAGE_KEY = 'activeTigger.context';

const storedContext = localStorage.getItem(CONTEXT_LOCAL_STORAGE_KEY);

export const defaultContext: AppContextValue = storedContext
  ? JSON.parse(storedContext)
  : {
      notifications: [],
      selectionConfig: {
        mode: 'deterministic',
        sample: 'untagged',
        displayPrediction: false,
        frameSelection: false,
        frame: null,
      },
      history: [],
      freqRefreshSimpleModel: 10,
    };

export type AppContextType = {
  appContext: AppContextValue;
  setAppContext: React.Dispatch<React.SetStateAction<AppContextValue>>;
};

export const AppContext = createContext<AppContextType>(null as unknown as AppContextType);

const _useAppContext = () => {
  const [appContext, setAppContext] = useState<AppContextValue>(defaultContext);

  //store context in localstorage
  useEffect(() => {
    localStorage.setItem(CONTEXT_LOCAL_STORAGE_KEY, JSON.stringify(appContext));
  }, [appContext]);

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
