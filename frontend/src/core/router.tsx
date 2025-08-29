import { Outlet, createHashRouter } from 'react-router-dom';

import { CurrentProjectState } from '../components/CurrentProjectState';
import { APIMiddlewares } from '../components/auth/APIMiddlewares';
import { AccountPage } from '../pages/AccountPage';
import { DocPage } from '../pages/DocPage';
import { CuratePage } from '../pages/ProjectCurationPage';
/*import { ExperimentalPage } from '../components/pages/ExperimentalPage';*/
import { RoleSelector } from '../core/RoleSelector';
import { BertopicPage } from '../pages/BertopicPage';
import { FinetunePage } from '../pages/FinetunePage';
import { HomePage } from '../pages/HomePage';
import { LoginPage } from '../pages/LoginPage';
import { MonitorPage } from '../pages/MonitorPage';
import { NotFound } from '../pages/NotFoundPage';
import { ProjectExportPage } from '../pages/ProjectExportPage';
import { GenPage } from '../pages/ProjectGenPage';
import { ProjectNewPage } from '../pages/ProjectNewPage';
import { ProjectPage } from '../pages/ProjectPage';
import { ProjectTagPage } from '../pages/ProjectTagPage';
import { ProjectsPage } from '../pages/ProjectsPage';
import { UsersPage } from '../pages/UsersPage';

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
    /*{
      path: '/experimental',
      element: (
        <APIMiddlewares>
          <ExperimentalPage />
        </APIMiddlewares>
      ),
    },*/
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
          path: '/projects/:projectName/explore/',
          element: (
            <>
              <BertopicPage />
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
