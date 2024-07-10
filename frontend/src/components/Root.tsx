import { FC, useMemo } from 'react';
import { RouterProvider } from 'react-router-dom';

import { AuthProvider } from '../core/auth';
import { getRouter } from '../core/router';

const Root: FC = () => {
  const router = useMemo(() => getRouter(), []);

  return (
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  );
};

export default Root;
