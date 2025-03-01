import { Outlet, createHashRouter } from 'react-router-dom';

import { CurrentProjectMonitoring } from '../components/CurrentProjectMonitoring';
import { AuthRequired } from '../components/auth/AuthRequired';
import { AccountPage } from '../components/pages/AccountPage';
import { CuratePage } from '../components/pages/CurationPage';
import { DocPage } from '../components/pages/DocPage';
import { HomePage } from '../components/pages/HomePage';
import { LoginPage } from '../components/pages/LoginPage';
import { MonitorPage } from '../components/pages/MonitorPage';
import { NotFound } from '../components/pages/NotFoundPage';
import { ProjectPredictPage } from '../components/pages/PredictPage';
import { ProjectAnnotationPage } from '../components/pages/ProjectAnnotationPage';
import { ProjectExplorePage } from '../components/pages/ProjectExplorePage';
import { ProjectExportPage } from '../components/pages/ProjectExportPage';
import { GenPage } from '../components/pages/ProjectGenPage';
import { ProjectNewPage } from '../components/pages/ProjectNewPage';
import { ProjectPage } from '../components/pages/ProjectPage';
import { ProjectPreparePage } from '../components/pages/ProjectPreparePage';
import { ProjectTestPage } from '../components/pages/ProjectTestPage';
import { ProjectsPage } from '../components/pages/ProjectsPage';
import { TrainPage } from '../components/pages/TrainPage';
import { UsersPage } from '../components/pages/UsersPage';
import { RoleSelector } from '../core/RoleSelector';

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
      path: '/projects/',
      element: (
        <AuthRequired>
          <ProjectsPage />
        </AuthRequired>
      ),
    },
    {
      path: '/projects/new',
      element: (
        <AuthRequired>
          <ProjectNewPage />
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
          element: (
            <>
              <RoleSelector allowedRoles={['manager', 'root']} />
              <ProjectPreparePage />
            </>
          ),
        },
        {
          path: '/projects/:projectName/explore',
          element: (
            <>
              <RoleSelector allowedRoles={['manager', 'root']} />
              <ProjectExplorePage />
            </>
          ),
        },
        {
          path: '/projects/:projectName/annotate/',
          element: (
            <>
              <ProjectAnnotationPage />
            </>
          ),
        },
        {
          path: '/projects/:projectName/curate/',
          element: (
            <>
              <RoleSelector allowedRoles={['manager', 'root']} />
              <CuratePage />
            </>
          ),
        },
        {
          path: '/projects/:projectName/generate/',
          element: (
            <>
              <RoleSelector allowedRoles={['manager', 'root']} />
              <GenPage />
            </>
          ),
        },
        {
          path: '/projects/:projectName/train/',
          element: (
            <>
              <RoleSelector allowedRoles={['manager', 'root']} />
              <TrainPage />
            </>
          ),
        },
        {
          path: '/projects/:projectName/test/',
          element: (
            <>
              <RoleSelector allowedRoles={['manager', 'root']} />
              <ProjectTestPage />
            </>
          ),
        },
        {
          path: '/projects/:projectName/predict/',
          element: (
            <>
              <RoleSelector allowedRoles={['manager', 'root']} />
              <ProjectPredictPage />
            </>
          ),
        },
        {
          path: '/projects/:projectName/export',
          element: (
            <>
              <RoleSelector allowedRoles={['manager', 'root']} />
              <ProjectExportPage />
            </>
          ),
        },
      ],
    },
  ]);
}
