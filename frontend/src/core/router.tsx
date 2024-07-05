import { createBrowserRouter } from 'react-router-dom';

import { HomePage } from '../components/HomePage';
import { LoginPage } from '../components/LoginPage';
import { AppContextValue } from './context';

export function getRouter(_appContext: AppContextValue) {
  return createBrowserRouter([
    {
      path: '/',
      element: <HomePage />,
    },
    {
      path: '/login',
      element: <LoginPage />,
    },
    // example with param an loader
    // {
    //   path: '/project/:projectid,
    //   element: <ProjectPage />,
    //   loader: projectLoader,
    // },);
  ]);
}
