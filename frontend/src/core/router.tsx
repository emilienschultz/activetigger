import { values } from 'lodash';
import { RouteObject, createBrowserRouter } from 'react-router-dom';

import { HomePage } from '../components/HomePage';
import { AppContextType } from './context';

export const pages: Record<'home', RouteObject> = {
  home: {
    path: '/',
    element: <HomePage />,
  },
  // example with param an loader
  // {
  //   path: '/project/:projectid,
  //   element: <ProjectPage />,
  //   loader: projectLoader,
  // },
};

export type PAGE_KEYS = keyof typeof pages;

export function getRouter(_ctx: AppContextType) {
  return createBrowserRouter(values(pages));
}
