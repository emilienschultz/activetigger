import { Outlet, createHashRouter } from 'react-router-dom';

import { CurrentProjectState } from '../components/CurrentProjectState';
import { APIMiddlewares } from '../components/auth/APIMiddlewares';
import { AccountPage } from '../components/pages/AccountPage';
import { CuratePage } from '../components/pages/CurationPage';
import { DocPage } from '../components/pages/DocPage';
import { ExperimentalPage } from '../components/pages/ExperimentalPage';
import { FinetunePage } from '../components/pages/FinetunePage';
import { HomePage } from '../components/pages/HomePage';
import { LoginPage } from '../components/pages/LoginPage';
import { MonitorPage } from '../components/pages/MonitorPage';
import { NotFound } from '../components/pages/NotFoundPage';
import { ProjectExportPage } from '../components/pages/ProjectExportPage';
import { GenPage } from '../components/pages/ProjectGenPage';
import { ProjectNewPage } from '../components/pages/ProjectNewPage';
import { ProjectPage } from '../components/pages/ProjectPage';
import { ProjectTagPage } from '../components/pages/ProjectTagPage';
import { ProjectsPage } from '../components/pages/ProjectsPage';
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
        <APIMiddlewares>
          <UsersPage />
        </APIMiddlewares>
      ),
    },
    {
      path: '/monitor',
      element: (
        <APIMiddlewares>
          <MonitorPage />
        </APIMiddlewares>
      ),
    },
    {
      path: '/experimental',
      element: (
        <APIMiddlewares>
          <ExperimentalPage />
        </APIMiddlewares>
      ),
    },
    {
      path: '/projects/',
      element: (
        <APIMiddlewares>
          <ProjectsPage />
        </APIMiddlewares>
      ),
    },
    {
      path: '/projects/new',
      element: (
        <APIMiddlewares>
          <ProjectNewPage />
        </APIMiddlewares>
      ),
    },
    {
      path: '/projects/:projectName',
      element: (
        <APIMiddlewares>
          <CurrentProjectState />
          <Outlet />
        </APIMiddlewares>
      ),
      children: [
        {
          path: '/projects/:projectName/',
          element: <ProjectPage />,
        },
        {
          path: '/projects/:projectName/tag/:elementId',
          element: <ProjectTagPage />,
        },
        {
          path: '/projects/:projectName/tag/',
          element: (
            <>
              <ProjectTagPage />
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
          path: '/projects/:projectName/finetune/',
          element: (
            <>
              <RoleSelector allowedRoles={['manager', 'root']} />
              <FinetunePage />
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
