import { FC, useEffect, useMemo, useState } from 'react';
import { RouterProvider } from 'react-router-dom';

import { AppContext, AppContextValue, defaultContext } from '../core/context';
import { getRouter } from '../core/router';

const Root: FC = () => {
  const storedAppContext = localStorage.getItem('appContext');

  const [appContext, setAppContext] = useState<AppContextValue>(
    storedAppContext ? JSON.parse(storedAppContext) : defaultContext,
  );

  useEffect(() => {
    localStorage.setItem('appContext', JSON.stringify(appContext));
  }, [appContext]);

  const router = useMemo(() => getRouter(appContext), [appContext]);

  return (
    <AppContext.Provider value={{ appContext: appContext, setAppContext }}>
      <RouterProvider router={router} />
    </AppContext.Provider>
  );
};

export default Root;
