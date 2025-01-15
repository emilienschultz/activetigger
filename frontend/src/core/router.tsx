import { Outlet, createHashRouter } from 'react-router-dom';

import { CurrentProjectMonitoring } from '../components/CurrentProjectMonitoring';
import { DocPage } from '../components/DocPage';
import { HomePage } from '../components/HomePage';
import { NotFound } from '../components/NotFoundPage';
import { ProjectAnnotationPage } from '../components/ProjectAnnotationPage';
import { ProjectExplorePage } from '../components/ProjectExplorePage';
import { ProjectExportPage } from '../components/ProjectExportPage';
import { GenPage } from '../components/ProjectGenPage';
import { ProjectNewPage } from '../components/ProjectNewPage';
import { ProjectPage } from '../components/ProjectPage';
import { ProjectPreparePage } from '../components/ProjectPreparePage';
import { ProjectTestPage } from '../components/ProjectTestPage';
import { ProjectTrainPage } from '../components/ProjectTrainPage';
import { ProjectsPage } from '../components/ProjectsPage';
import { AuthRequired } from '../components/auth/AuthRequired';
import { AccountPage } from '../components/pages/AccountPage';
import { CuratePage } from '../components/pages/CurationPage';
import { LoginPage } from '../components/pages/LoginPage';
import { MonitorPage } from '../components/pages/MonitorPage';
import { UsersPage } from '../components/pages/UsersPage';

export function getRouter() {
  return createHashRouter([
    {
      path: '*',
      element: <NotFound />,
    },
    {
      path: '/',
      element: <HomePage />,
    },
    {
      path: '/login',
      element: <LoginPage />,
    },
    {
      path: '/account',
      element: <AccountPage />,
    },
    {
      path: '/help',
      element: <DocPage />,
    },
    {
      path: '/users',
      element: (
        <AuthRequired>
          <UsersPage />
        </AuthRequired>
      ),
    },
    {
      path: '/monitor',
      element: (
        <AuthRequired>
          <MonitorPage />
        </AuthRequired>
      ),
    },
    {
      path: '/projects/new',
      element: (
        //AuthRequired makes sure that the user is currently authenticated before rendering this route page
        <AuthRequired>
          <ProjectNewPage />
        </AuthRequired>
      ),
    },
    {
      path: '/projects/',
      element: (
        <AuthRequired>
          <ProjectsPage />
        </AuthRequired>
      ),
    },
    {
      path: '/projects/:projectName',
      element: (
        <AuthRequired>
          <CurrentProjectMonitoring />
          <Outlet />
        </AuthRequired>
      ),
      children: [
        {
          path: '/projects/:projectName/',
          element: <ProjectPage />,
        },
        {
          path: '/projects/:projectName/annotate/:elementId',
          element: <ProjectAnnotationPage />,
        },
        {
          path: '/projects/:projectName/prepare/',
          element: <ProjectPreparePage />,
        },
        {
          path: '/projects/:projectName/explore',
          element: <ProjectExplorePage />,
        },
        {
          path: '/projects/:projectName/annotate/',
          element: <ProjectAnnotationPage />,
        },
        {
          path: '/projects/:projectName/curate/',
          element: <CuratePage />,
        },
        {
          path: '/projects/:projectName/generate/',
          element: <GenPage />,
        },
        {
          path: '/projects/:projectName/train/',
          element: <ProjectTrainPage />,
        },
        {
          path: '/projects/:projectName/test/',
          element: <ProjectTestPage />,
        },
        {
          path: '/projects/:projectName/export',
          element: <ProjectExportPage />,
        },
      ],
    },
  ]);
}
