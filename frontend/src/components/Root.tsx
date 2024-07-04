import { FC, useMemo } from 'react';
import { RouterProvider } from 'react-router-dom';

import { AppContext } from '../core/context';
import { getRouter } from '../core/router';

const Root: FC = () => {
  const appContextValue = useMemo(
    () => ({
      // TODO
    }),
    [],
  );

  const router = useMemo(() => getRouter(appContextValue), [appContextValue]);

  return (
    <AppContext.Provider value={appContextValue}>
      <RouterProvider router={router} />
    </AppContext.Provider>
  );
};

export default Root;
