import { Outlet, createHashRouter } from 'react-router-dom';

import { CurrentProjectMonitoring } from '../components/CurrentProjectMonitoring';
import { HelpPage } from '../components/HelpPage';
import { HomePage } from '../components/HomePage';
import { LoginPage } from '../components/LoginPage';
import { NotFound } from '../components/NotFoundPage';
import { ProjectAnnotationPage } from '../components/ProjectAnnotationPage';
import { ProjectExplorationPage } from '../components/ProjectExplorationPage';
import { ProjectExportPage } from '../components/ProjectExportPage';
import { ProjectFeaturesPage } from '../components/ProjectFeaturesPage';
import { ProjectNewPage } from '../components/ProjectNewPage';
import { ProjectPage } from '../components/ProjectPage';
import { ProjectTestPage } from '../components/ProjectTestPage';
import { ProjectTrainPage } from '../components/ProjectTrainPage';

import { ProjectsPage } from '../components/ProjectsPage';
import { UsersPage } from '../components/UsersPage';

import { AuthRequired } from '../components/auth/AuthRequired';

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
      path: '/help',
      element: <HelpPage />,
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
          path: '/projects/:projectName/features/',
          element: <ProjectFeaturesPage />,
        },
        {
          path: '/projects/:projectName/explorate',
          element: <ProjectExplorationPage />,
        },
        {
          path: '/projects/:projectName/annotate/',
          element: <ProjectAnnotationPage />,
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
