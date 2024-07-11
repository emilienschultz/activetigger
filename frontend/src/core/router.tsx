import { createBrowserRouter } from 'react-router-dom';

import { HelpPage } from '../components/HelpPage';
import { HomePage } from '../components/HomePage';
import { LoginPage } from '../components/LoginPage';
import { ProjectNewPage } from '../components/ProjectNewPage';
import { ProjectPage } from '../components/ProjectPage';
import { ProjectsPage } from '../components/ProjectsPage';
import { AuthRequired } from '../components/auth/AuthRequired';

export function getRouter() {
  return createBrowserRouter([
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
      path: '/projects/new',
      element: (
        //AuthRequired makes sure that the user is currently authenticated before rendering this route page
        <AuthRequired>
          <ProjectNewPage />
        </AuthRequired>
      ),
    },
    {
      path: '/projects',
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
          <ProjectPage />
        </AuthRequired>
      ),
    },
  ]);
}
